import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
from typing import Optional, Tuple



# =====================
# Colors
# =====================

def hex_to_rgb(hex_color):
    """
    Konvertiert einen Hexadezimal-Farbcode in ein RGB-Tupel.

    Die Funktion entfernt zuerst das führende '#' vom Hexadezimal-String, falls es vorhanden ist.
    Anschließend wird der String in drei Teile aufgeteilt (jeweils zwei Zeichen), die jeweils
    die rote, grüne und blaue Komponente repräsentieren. Diese Teile werden dann von Hexadezimal
    in ganze Zahlen konvertiert, um die RGB-Werte zu erhalten.

    Args:
        hex_color (str): Der Hexadezimal-String des Farbcodes. Kann mit oder ohne führendes '#' sein.

    Returns:
        tuple: Ein Tuple, das die RGB-Komponenten der Farbe enthält (jeweils als ganze Zahl von 0 bis 255).

    Beispiel:
        >>> hex_to_rgb("#FFFFFF")
        (255, 255, 255)

        >>> hex_to_rgb("000000")
        (0, 0, 0)
    """

    hex_color = hex_color.lstrip('#')  # Entfernen des '#' Zeichens, falls vorhanden
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))  # Konvertieren von Hex zu RGB
    return rgb

def apply_colormap(mask: torch.Tensor, colormap) -> np.ndarray:
    """
    Wendet eine Farbkarte auf eine Maske an und konvertiert diese in ein NumPy-Array.

    Diese Funktion nimmt eine Maske als PyTorch-Tensor und wendet darauf eine Farbkarte an.
    Die resultierende farbige Maske wird dann in ein NumPy-Array umgewandelt, das für 
    Visualisierungszwecke verwendet werden kann. Die Farbkarte transformiert die Werte
    der Maske in Farben, wodurch die Maske visuell interpretierbar wird.

    Args:
        mask (torch.Tensor): Die Maske als PyTorch-Tensor.
        colormap: Eine Farbkarte, die auf die Maske angewendet werden soll.

    Returns:
        np.ndarray: Die farbige Maske als NumPy-Array.

    Die Maske wird zunächst in ein NumPy-Array umgewandelt, auf das die Farbkarte angewendet wird.
    Die resultierende farbige Maske wird in ein Array im Format [Höhe, Breite, 3] konvertiert,
    multipliziert mit 255, um den Wertebereich auf 0-255 zu skalieren, und in den Datentyp np.uint8 umgewandelt.
    """

    # Anwenden der Farbkarte auf die Maske und Konvertierung in ein 3-Kanal-NumPy-Array
    colored_mask = colormap(mask.numpy())[:, :, :3]

    # Skalieren der Werte auf den Bereich 0-255 und Umwandlung in den Datentyp np.uint8
    return (colored_mask * 255).astype(np.uint8)


# =====================
# Masks
# =====================

def make_2d_mask(mask):
    """
    Konvertiert einen gegebenen Masken-Tensor in einen 2D-Tensor.

    Diese Funktion nimmt einen Tensor, der eine Maske darstellt, und reduziert dessen Dimensionalität auf 2D.
    Sie behandelt unterschiedliche Eingangsformen, je nachdem, ob der Tensor 4D (z.B. Batch-Größe und Kanäle enthalten),
    3D (z.B. nur Kanäle enthalten) oder bereits 2D ist.

    Args:
        mask (torch.Tensor): Ein Tensor, der die Maske repräsentiert. Kann 2D, 3D oder 4D sein.

    Returns:
        torch.Tensor: Ein 2D-Tensor, der die Maske darstellt.

    Die Funktion überprüft zuerst die Dimensionalität des Eingangstensors. Wenn der Tensor 4D ist, wird erwartet,
    dass die ersten beiden Dimensionen die Batch-Größe und die Kanäle sind, die entfernt werden. Bei einem 3D-Tensor
    wird angenommen, dass die erste Dimension die Kanäle sind, die ebenfalls entfernt werden. Ist der Tensor bereits 2D,
    wird er unverändert zurückgegeben.
    """

    # Entferne zusätzliche Dimensionen, um einen 2D-Tensor zu erhalten
    if len(mask.shape) == 4:
        # Annahme: Tensorformat ist [Batch, Kanal, Höhe, Breite]
        return mask.squeeze(0).squeeze(0)  # Entferne die Batch- und Kanaldimension
    elif len(mask.shape) == 3:
        # Annahme: Tensorformat ist [Kanal, Höhe, Breite]
        return mask.squeeze(0)  # Entferne die Kanaldimension
    # Wenn der Tensor bereits 2D ist, gib ihn unverändert zurück
    return mask

def mask2cv(mask_np):
    """
    Konvertiert eine Maske im NumPy-Format in ein für OpenCV geeignetes Format.

    Diese Funktion nimmt eine Maske als NumPy-Array und konvertiert sie in ein Format,
    das von OpenCV verarbeitet werden kann. Die Eingabemaske sollte im Format [Höhe, Breite, Kanäle]
    vorliegen. Die Funktion skaliert die Werte der Maske in den Bereich von 0 bis 255, was dem 
    Standardformat für Bilder in OpenCV entspricht.

    Args:
        mask_np (numpy.ndarray): Die Eingabemaske als NumPy-Array. Sollte im Format [Höhe, Breite, Kanäle] sein.

    Returns:
        numpy.ndarray: Die konvertierte Maske im OpenCV-kompatiblen Format.

    Die Funktion wandelt zuerst den Datentyp der Maske in `np.float32` um, um eine korrekte Skalierung
    zu gewährleisten. Anschließend werden die Werte der Maske auf den Bereich von 0 bis 255 skaliert, 
    was dem Standardformat für Bilder in OpenCV entspricht.
    """

    msk_cv2 = mask_np.astype(np.float32)  # Umwandlung in float32 für korrekte Skalierung
    msk_cv2 = msk_cv2 * 255  # Skalierung der Werte auf den Bereich 0 bis 255
    return msk_cv2

def blur_cvmask(mask, blur_factor):
    """
    Wendet einen Gaußschen Weichzeichner (Blur) auf eine Maske an.

    Diese Funktion verwendet die Gaußsche Unschärfe von OpenCV, um eine gegebene Maske zu weichzeichnen.
    Die Intensität der Unschärfe wird durch den 'blur_factor' bestimmt. Ein größerer 'blur_factor' führt zu einer
    stärkeren Unschärfe. Um einen validen Kernel für die Gaußsche Unschärfe zu gewährleisten, wird der 'blur_factor'
    auf die nächste ungerade Zahl erhöht, falls er eine gerade Zahl ist.

    Args:
        mask (numpy.ndarray): Die Eingabemaske, die weichgezeichnet werden soll.
        blur_factor (int): Der Faktor, der die Stärke der Unschärfe bestimmt.

    Returns:
        numpy.ndarray: Die weichgezeichnete Maske.

    Ein gerader 'blur_factor' wird auf die nächste ungerade Zahl erhöht, um einen korrekten Kernel für die
    Gaußsche Unschärfe zu gewährleisten. Die Maske wird dann mit diesem Kernel weichgezeichnet.
    """

    # Stelle sicher, dass der Blur-Faktor ungerade ist
    if blur_factor % 2 == 0:  
        blur_factor += 1  

    # Anwenden der Gaußschen Unschärfe auf die Maske
    blured_mask = cv2.GaussianBlur(mask, (blur_factor, blur_factor), 15)
    return blured_mask

def adjust_cvmask_size(mask, factor):
    """
    Ändert die Größe einer Maske durch Schrumpfen oder Wachsen abhängig vom Faktor.

    Diese Funktion führt eine morphologische Erosion oder Dilatation auf einer Maske aus.
    Bei einem positiven Faktor wird die Maske erweitert (Dilatation), bei einem negativen Faktor
    wird sie verkleinert (Erosion). Die Intensität der Operation wird durch den absoluten Wert 
    des Faktors bestimmt.

    Args:
        mask (numpy.ndarray): Eine Maske, die bearbeitet werden soll.
        factor (int): Der Faktor, der die Intensität und Richtung der Größenänderung bestimmt.
                      Positive Werte führen zur Dilatation, negative zur Erosion.

    Returns:
        numpy.ndarray: Die bearbeitete Maske.

    Ein positiver Faktor führt zur Dilatation (Vergrößerung), während ein negativer Faktor 
    zur Erosion (Verkleinerung) der Maske führt. Bei einem Faktor von 0 bleibt die Maske unverändert.
    """

    if factor > 0:
        # Dilatation (Vergrößern der Maske)
        mask = cv2.dilate(mask, kernel=np.ones((3, 3), np.uint8), iterations=abs(factor))
    elif factor < 0:
        # Erosion (Verkleinern der Maske)
        mask = cv2.erode(mask, kernel=np.ones((3, 3), np.uint8), iterations=abs(factor))
    else:
        # Keine Änderung, wenn der Faktor 0 ist
        pass

    return mask

def dilate_mask(mask: torch.Tensor, dilation_factor: float) -> torch.Tensor:
    """
    Führt eine Dilatation (Erweiterung) auf einer gegebenen Maske mit einem spezifischen Faktor durch.

    Diese Funktion verwendet die Dilatationstechnik von OpenCV, um die Strukturen innerhalb einer Maske zu erweitern.
    Der Dilatationsfaktor bestimmt die Größe des quadratischen Kernels, der für die Dilatation verwendet wird.

    Args:
        mask (torch.Tensor): Die Eingabemaske als PyTorch-Tensor.
        dilation_factor (float): Der Faktor, der die Größe des quadratischen Kernels bestimmt.
                                Größere Werte führen zu einer stärkeren Dilatation.

    Returns:
        torch.Tensor: Die dilatierte Maske als PyTorch-Tensor.

    Der Kernel für die Dilatation wird basierend auf dem 'dilation_factor' erstellt. Ein größerer Faktor
    bewirkt eine stärkere Erweiterung der Maske. Die Funktion konvertiert die Maske zunächst in ein NumPy-Array,
    führt die Dilatation durch und konvertiert das Ergebnis zurück in einen PyTorch-Tensor.
    """

    # Berechnung der Kernelgröße basierend auf dem Dilatationsfaktor
    kernel_size = int(dilation_factor * 2) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # Durchführen der Dilatation auf die Maske
    mask_dilated = cv2.dilate(mask.numpy(), kernel, iterations=1)

    # Rückgabe der dilatierten Maske als PyTorch-Tensor
    return torch.from_numpy(mask_dilated)
# =====================
# Image Processing 
# =====================

def resize_image(image: np.ndarray, dimensions: Tuple[int, int]) -> np.ndarray:
    """
    Skaliert ein Bild auf die gegebenen Dimensionen unter Verwendung linearer Interpolation.

    Diese Funktion nimmt ein Bild im NumPy-Array-Format und ändert seine Größe auf die spezifizierten
    Dimensionen. Dabei wird lineare Interpolation verwendet, um die Pixelwerte des skalierten Bildes
    zu berechnen. Lineare Interpolation ist eine gängige Methode für die Größenänderung von Bildern,
    die ein gutes Gleichgewicht zwischen Qualität und Rechenleistung bietet.

    Args:
        image (np.ndarray): Das Originalbild als NumPy-Array.
        dimensions (Tuple[int, int]): Ein Tupel, das die Zielgröße des Bildes angibt (Breite, Höhe).

    Returns:
        np.ndarray: Das skalierte Bild als NumPy-Array.

    Das Bild wird mit der Funktion 'cv2.resize' der OpenCV-Bibliothek skaliert. Der Parameter
    'interpolation=cv2.INTER_LINEAR' gibt an, dass lineare Interpolation für das Skalieren verwendet wird.
    """

    # Skalieren des Bildes auf die angegebenen Dimensionen
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_LINEAR)

def split_image_mask(image, device):
    """
    Trennt ein PIL-Bild in seine RGB-Komponenten und eine Alpha-Maske.

    Die Funktion konvertiert ein PIL-Bild zuerst in den RGB-Farbraum. Dann wird das RGB-Bild und die
    Alpha-Maske (falls vorhanden) in NumPy-Arrays umgewandelt, normalisiert und schließlich in PyTorch-Tensoren 
    konvertiert. Falls keine Alpha-Maske vorhanden ist, wird eine leere Maske erstellt.

    Args:
        image (PIL.Image): Das PIL-Bild, das sowohl Farb- als auch Maskeninformationen enthält.
        device (torch.device): Das Gerät, auf dem die Berechnung ausgeführt wird (z.B. 'cpu' oder 'cuda').

    Returns:
        tuple: Ein Tuple, das zwei Elemente enthält:
               - Ein Tensor, der das RGB-Bild repräsentiert.
               - Ein Tensor, der die Alpha-Maske repräsentiert (oder eine leere Maske, falls keine Alpha-Maske vorhanden ist).

    Beispiel:
        >>> from PIL import Image
        >>> image = Image.open('bild_mit_alpha.png')
        >>> image_rgb, mask = split_image_mask(image, torch.device('cuda'))
    """

    # Konvertiere das Bild in RGB und dann in einen Tensor
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,].to(device)

    # Extrahiere die Alpha-Maske, falls vorhanden
    if 'A' in image.getbands():
        mask = np.array(image.getchannel('A')).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,].to(device)
    else:
        # Erstelle eine leere Maske, falls keine Alpha-Maske vorhanden ist
        mask = torch.zeros((64, 64), dtype=torch.float32, device=device)

    return (image_rgb, mask)


def blend_rgba_with_background(rgba_tensor, bg_color_tensor):
    """
    Kombiniert ein RGBA-Bild mit einem RGB-Hintergrund und gibt ein RGB-Bild zurück.

    :param rgba_tensor: Ein RGBA-Bildtensor der Form [1, Höhe, Breite, 4].
    :param bg_color_tensor: Ein RGB-Hintergrundtensor der Form [Höhe, Breite, 3].
    :return: Ein RGB-Bildtensor der Form [1, Höhe, Breite, 3].
    """
    # Extrahiere den RGB-Teil und den Alpha-Kanal des RGBA-Bildes
    rgb = rgba_tensor[:, :, :, :3]
    alpha = rgba_tensor[:, :, :, 3:]#.to(dtype=torch.float32) / 255.0  # Normalisiere den Alpha-Kanal

    # Stelle sicher, dass der Hintergrundfarbentensor die richtige Form hat
    bg_color_tensor = bg_color_tensor.view(1, 1, 1, 3).expand_as(rgb)

    # Führe das Alpha-Blending durch
    # Hier wird angenommen, dass bg_color_tensor bereits auf dem richtigen Gerät ist
    blended_rgb = (alpha * rgb) + ((1 - alpha) * bg_color_tensor)

    return blended_rgb


def enhance_edges(image_np_rgb, alpha=1.5, beta=50, edge_alpha=1.0):
    """
    Erhöht den Kontrast, die Helligkeit und die Kantenstärke des Bildes.

    :param image_np_rgb: Eingabebild in NumPy-Array-Form.
    :param alpha: Faktor für den Kontrast.
        Wertebereich: Typischerweise zwischen 1.0 und 3.0.
        Standardwert: 1.0 (keine Änderung des Kontrasts).
        Hinweis: Werte größer als 1 erhöhen den Kontrast, während Werte zwischen 0 und 1 ihn verringern. Extrem hohe Werte können zu einer Sättigung führen, bei der Details verloren gehen.
    :param beta: Wert für die Helligkeit.
        Wertebereich: Kann positiv oder negativ sein, typischerweise zwischen -100 und 100.
        Standardwert: 0 (keine Änderung der Helligkeit).
        Hinweis: Positive Werte erhöhen die Helligkeit, negative Werte verringern sie. Zu hohe oder zu niedrige Werte können dazu führen, dass helle oder dunkle Bereiche keine Details mehr aufweisen.
    :param edge_alpha: Faktor für die Kantenstärke.
        Wertebereich: Normalerweise zwischen 0 und 1.
        Standardwert: 1.0 (volle Stärke der Kanten).
        Hinweis: Ein Wert von 0 würde keine Kanten hinzufügen, während ein Wert von 1 die Kanten deutlich hervorhebt. Wenn die Kanten zu stark hervorgehoben werden, kann das Bild überladen wirken und die Segmentierung beeinträchtigen.
    :return: Angepasstes Bild als NumPy-Array.
    """
    # Adjust contrast and brightness
    adjusted_image = np.clip(alpha * image_np_rgb.astype(np.float32) + beta, 0, 255).astype(np.uint8)
    
    # Convert to grayscale
    gray = cv2.cvtColor(adjusted_image, cv2.COLOR_RGB2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 100, 200)
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
    
    # Combine original image with edges
    enhanced_image = cv2.addWeighted(adjusted_image, 1, edges_colored, edge_alpha, 0)

    return enhanced_image

def img_combine_mask_rgba(image_np_rgb, msk_cv2_blurred):
    """
    Kombiniert ein RGB-Bild und eine geblurte Maske in ein einziges RGBA-Bild (als NumPy-Array).

    Diese Funktion nimmt ein RGB-Bild und eine geblurte Maske (beide als NumPy-Arrays) und fügt sie zu einem
    einzigen RGBA-Bild zusammen, wobei die Maske als Alpha-Kanal verwendet wird.

    Args:
        image_np_rgb (numpy.ndarray): Ein RGB-Bild als NumPy-Array im Format [Höhe, Breite, 3].
        msk_cv2_blurred (numpy.ndarray): Eine geblurte Maske als NumPy-Array im Format [Höhe, Breite, 1].

    Returns:
        numpy.ndarray: Das kombinierte Bild im RGBA-Format als NumPy-Array.

    Das RGB-Bild und die geblurte Maske werden entlang der letzten Dimension (Kanäle) miteinander verbunden,
    wobei die Maske als Alpha-Kanal für Transparenz hinzugefügt wird.
    """

    # Kombiniere das RGB-Bild und die geblurte Maske zu einem RGBA-Bild
    image_with_alpha = np.concatenate((image_np_rgb, msk_cv2_blurred), axis=2)#.astype(np.uint8)
    return image_with_alpha

def overlay_image(background: np.ndarray, foreground: np.ndarray, alpha: float) -> np.ndarray:
    """
    Legt ein Vordergrundbild mit einer bestimmten Deckkraft (Alpha) über ein Hintergrundbild.

    Diese Funktion nimmt zwei Bilder (Hintergrund und Vordergrund) sowie einen Alphawert für die Deckkraft
    des Vordergrundbildes. Das Vordergrundbild wird mit dieser Deckkraft auf das Hintergrundbild gelegt,
    wobei die beiden Bilder kombiniert und die Deckkraft des Vordergrundbildes berücksichtigt wird.

    Args:
        background (np.ndarray): Das Hintergrundbild als NumPy-Array.
        foreground (np.ndarray): Das Vordergrundbild als NumPy-Array.
        alpha (float): Die Deckkraft des Vordergrundbildes (Wert zwischen 0 und 1).

    Returns:
        np.ndarray: Das resultierende Bild nach der Überlagerung.

    Die Funktion verwendet 'cv2.addWeighted' zur Kombination der beiden Bilder. Der Alphawert bestimmt,
    wie stark das Vordergrundbild im Verhältnis zum Hintergrundbild sichtbar ist. Ein Alpha-Wert von 1 
    würde bedeuten, dass nur das Vordergrundbild sichtbar ist, während ein Wert von 0 nur das Hintergrundbild
    zeigt.
    """

    # Kombinieren der Bilder unter Berücksichtigung der Deckkraft (Alpha)
    return cv2.addWeighted(background, 1 - alpha, foreground, alpha, 0)

# =====================
# Debug Image Generation
# =====================

def createDebugImage( image, dino_bbox, dino_pharses, dino_logits, toggle_sam_debug=False, sam_contrasts_helper=1.25, sam_brightness_helper=0, sam_hint_grid_points=None, sam_hint_grid_labels=None):
    """
    Erstellt ein Debug-Bild, das Bounding-Boxen, Phrasen, Logits und optionale Gitterpunkte anzeigt.

    Args:
        image (PIL.Image): Das ursprüngliche Bild.
        dino_bbox (list): Liste von Bounding-Box-Koordinaten.
        dino_phrases (list): Liste von Phrasen, die mit den Bounding-Boxen korrespondieren.
        dino_logits (list): Liste von Logit-Werten für jede Bounding-Box.
        toggle_sam_debug (bool): Schalter, um zusätzliche Debug-Informationen anzuzeigen.
        sam_contrasts_helper (float): Kontrasthilfe für das Bild.
        sam_brightness_helper (float): Helligkeitshilfe für das Bild.
        sam_hint_grid_points (list): Liste von Gitterpunkten für Debug-Zwecke.
        sam_hint_grid_labels (list): Liste von Labels für die Gitterpunkte.

    Returns:
        torch.Tensor: Ein Tensor des bearbeiteten Debug-Bildes.

    Diese Funktion nimmt ein Bild und verschiedene Debug-Informationen entgegen und zeichnet
    Bounding-Boxen, Text und optionale Gitterpunkte auf das Bild. Das Ergebnis wird als Tensor
    zurückgegeben, der weiterverarbeitet oder angezeigt werden kann.
    """
    
    width, height = image.size
    debug_image = image.copy().convert('RGB')
    debug_image_np = np.array(debug_image)

    if toggle_sam_debug:
        # enhance_edges()
        debug_image_np = enhance_edges(debug_image_np, alpha=sam_contrasts_helper, beta=sam_brightness_helper, edge_alpha=1.0)
        debug_image = Image.fromarray(debug_image_np)

        if sam_hint_grid_points != [None] and sam_hint_grid_points != [None]: 

            # Erstellen einer transparenten Ebene
            transparent_layer = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            draw = ImageDraw.Draw(debug_image)

            for index, bbox in enumerate(dino_bbox):
                x_min, y_min, x_max, y_max = map(int, bbox)
                box_width = x_max - x_min
                box_height = y_max - y_min

                # Definieren Sie die Schrittgrößen für die Grid-Erstellung
                x_step = max(3, int(box_width / 20))
                y_step = max(3, int(box_height / 20))

                # Erstellen Sie Punkte innerhalb der Bounding Box
                grid_points = []
                for y in range(y_min, y_max, y_step):
                    for x in range(x_min, x_max, x_step):
                        grid_points.append((x, y))
                    if sam_hint_grid_labels[index] is not None:
                        for point, label in zip(grid_points, sam_hint_grid_labels[index]):
                            x, y = point
                            if label > 0:
                                color = (0, 255, 0, 128)  # Halbtransparentes Grün
                            else:
                                color = (255, 0, 0, 128)  # Halbtransparentes Rot
                            # Zeichnen eines halbtransparenten Kreises
                            draw.ellipse((x-2, y-2, x+2, y+2), fill=color)
            # Kombinieren der transparenten Ebene mit dem Originalbild
            image_with_transparent_layer = Image.alpha_composite(debug_image.convert('RGBA'), transparent_layer)
            debug_image = image_with_transparent_layer
        # Anzeigen des Bildes
        #image_with_transparent_layer.show()
    else:
        debug_image = Image.fromarray(debug_image_np)

    draw = ImageDraw.Draw(debug_image)
    text_padding = 2.5  # Anpassbares Padding
    for bbox, conf, phrase in zip(dino_bbox, dino_logits, dino_pharses):
        bbox = tuple(map(int, bbox))
        draw.rectangle(bbox, outline="red", width=2)
        
        # Erstelle das Rechteck für den Hintergrund des Textes mit Padding
        text_bg_x1 = bbox[0]
        text_bg_y1 = bbox[1]
        text_bg_x2 = bbox[0] + 120  # Beispielbreite des Hintergrunds
        text_bg_y2 = bbox[1] + 20   # Beispielhöhe des Hintergrunds
        draw.rectangle(((text_bg_x1, text_bg_y1), (text_bg_x2, text_bg_y2)), fill="red")  # Hintergrund für den Text

        # Textkoordinaten mit Padding
        text_x = bbox[0] + text_padding
        text_y = bbox[1] + text_padding

        draw.text((text_x, text_y), f"{phrase}: {conf:.2f}", fill="white")  # Weißer Text auf rotem Hintergrund

    transform = transforms.ToTensor()
    tensor_image = transform(debug_image)
    tensor_image_expanded = tensor_image.unsqueeze(0)
    tensor_image_formated = tensor_image_expanded.permute(0, 2, 3, 1)
    
    return tensor_image_formated




