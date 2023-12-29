/**
 * 
 *  TODO - Figure out how to connect to dynamic Python outputs. 
 * 
 */


import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

let res = await api.fetchApi('/impact/wildcards/list');
console.log(res)

function updateWidgetPositions(node) {
  const NODE_TITLE_HEIGHT = 30; // Die Höhe der Titelleiste der Node
  const NODE_SLOT_HEIGHT = 20; // Die Höhe eines einzelnen Slots
  const NODE_WIDGET_HEIGHT = 20; // Die Höhe eines einzelnen Widgets
  const spacing = 5; // Der Abstand zwischen Widgets

  let yOffset = (node.outputs.length * LiteGraph.NODE_SLOT_HEIGHT) + spacing;
  for (let i = 0; i < node.widgets.length; i++) {
    let widget = node.widgets[i];
    if (widget != null && widget.y != null) {
      widget.y = yOffset;
      yOffset += (widget.height ||  LiteGraph.NODE_SLOT_HEIGHT) + spacing;
    }
  }
  node.size[1] = yOffset + 30;
  node.setDirtyCanvas(true, true);
}


api.addEventListener('dnl13_dino_phrases', async (event) => {
  console.log("addEventListener event", event)
  const n = app.graph.getNodeById(+(event.detail.unique_id || app.runningNodeId));
  if (!n) return;

  console.log("addEventListener node", n)

  let empty = []
  const defaults = empty.concat(event.detail.outputs.default);
  let detected = empty.concat(event.detail.outputs.detected);
  let newOutput = defaults.concat(detected);

  event.detail.outputs.detected.forEach(element => {
    n.addOutput(element.name, element.type)
  });
  n.outputs = newOutput

  const newNodeSize = n.computeSize();
  n.size = newNodeSize;

  updateWidgetPositions(n)
  n.setDirtyCanvas(true, true);
});





const dnl13_outputs = {
  name: 'dnl13.dynamic_outputs',

  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    if (!nodeData.name.endsWith('(dnl13)')) { return }
    switch (nodeData.name) {
      case 'DinoSegmentationProcessor (dnl13)': {

        const onConnectionsChange = nodeType.prototype.onConnectionsChange;
        nodeType.prototype.onConnectionsChange = function (type, index, connected, link_info) {
          console.log(type)
          console.log(index)
          console.log(connected)
          console.log(link_info)
          if (type == 2) {
            // connect output
            if (connected && index == 0) {
              if (nodeData.name == 'ImpactSwitch' && app.graph._nodes_by_id[link_info.target_id]?.type == 'Reroute') {
                app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
              }
  
              if (this.outputs[0].type == '*') {
                if (link_info.type == '*') {
                  app.graph._nodes_by_id[link_info.target_id].disconnectInput(link_info.target_slot);
                }
                else {
                  // propagate type
                  this.outputs[0].type = link_info.type;
                  this.outputs[0].label = link_info.type;
                  this.outputs[0].name = link_info.type;
  
                  for (let i in this.inputs) {
                    let input_i = this.inputs[i];
                    if (input_i.name != 'select' && input_i.name != 'sel_mode')
                      input_i.type = link_info.type;
                  }
                }
              }
            }
  
            return;
          }
            console.log(type)
            console.log(index)
            console.log(connected)
            console.log(link_info)
            if(!link_info || this.outputs[0].type != '*')
                   return;
            
          }
             

        /* for later */
        break
      }
      default: {
        break
      }
    }
  },
}

app.registerExtension(dnl13_outputs)