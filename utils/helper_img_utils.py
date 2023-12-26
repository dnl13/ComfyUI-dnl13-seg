import torch
import numpy as np
from PIL import Image, ImageDraw
import cv2
import torchvision.transforms.v2 as v2
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F






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






def blend_rgba_with_background(rgba_tensor, bg_color_tensor):
    """
    Kombiniert ein RGBA-Bild mit einem RGB-Hintergrund und gibt ein RGB-Bild zurück.

    :param rgba_tensor: Ein RGBA-Bildtensor der Form [1, Höhe, Breite, 4].
    :param bg_color_tensor: Ein RGB-Hintergrundtensor der Form [3, Höhe, Breite].
    :return: Ein RGB-Bildtensor der Form [1, Höhe, Breite, 3].
    """
    # Extrahieren Sie den RGB-Teil und den Alpha-Kanal des Bildes
    rgb = rgba_tensor[:, :, :, :3]
    alpha = rgba_tensor[:, :, :, 3:4].squeeze(0)  # Entfernen Sie die Batch-Dimension

    # Führen Sie das Alpha-Blending durch
    blended_rgb = alpha * rgb + (1 - alpha) * bg_color_tensor

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






def createDebugImage( image, dino_bbox, dino_pharses, dino_logits, toggle_sam_debug, sam_contrasts_helper, sam_brightness_helper, sam_hint_grid_points, sam_hint_grid_labels):
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




