import {
  JupyterFrontEnd,
  JupyterFrontEndPlugin
} from '@jupyterlab/application';
import { INotebookTracker } from '@jupyterlab/notebook';
import { NotebookActions } from '@jupyterlab/notebook';
import { CodeCell } from '@jupyterlab/cells';

import { ISettingRegistry } from '@jupyterlab/settingregistry';

/**
 * Initialization data for the @shannon-shen/chapyter extension.
 */
const plugin: JupyterFrontEndPlugin<void> = {
  id: '@shannon-shen/chapyter:plugin',
  description: 'A Natural Language-Based Python Program Interpreter',
  autoStart: true,
  requires: [INotebookTracker],
  optional: [ISettingRegistry],
  activate: (app: JupyterFrontEnd, tracker: INotebookTracker) => {
    NotebookActions.executed.connect((sender, args) => {
      if (args.success && !args.cell.model.getMetadata('chatGen')) {
        console.log('Real Code cell was successfully executed');

        // It must be true that the cell is a code cell (otherwise it would not have been executed)
        let codeCell = args.cell as CodeCell;
        console.log("Executed cell:", codeCell);
        
        // We only want to automatically generate a new cell if the code cell starts with a magic command (e.g. %chat)
        let codeCellText = codeCell.model.sharedModel.getSource()
        if (codeCellText.startsWith('%chat') || codeCellText.startsWith('%%chat')) {

          // because it is successfully executed 
          let notebook = tracker.currentWidget;
          if (notebook) {
            
            NotebookActions.selectAbove(notebook.content);

            let newCell = notebook.content.activeCell as CodeCell;
            console.log('New cell:', newCell);
            if (newCell) {
              newCell.model.setMetadata("chatGen", true);
            } 

            NotebookActions.run(notebook.content, notebook.sessionContext);
            NotebookActions.hideCode(notebook.content);
            NotebookActions.selectBelow(notebook.content);
          }
        }
      }
    });
  }
};

export default plugin;
