import { app, BrowserWindow, ipcMain, dialog } from 'electron';
import * as path from 'path';
const contextMenu = require('electron-context-menu');

let mainWindow: BrowserWindow | null;

const template = [
  { label: 'Minimize', click: () => mainWindow?.minimize() },
  { label: 'Maximize', click: () => mainWindow?.maximize() },
  { type: 'separator' },
  { label: 'Copy', click: () => mainWindow?.webContents.copy() },
  { label: 'Paste', click: () => mainWindow?.webContents.paste() },
  { label: 'Delete', click: () => mainWindow?.webContents.delete() },
  { type: 'separator' },
  { type: 'separator' },
  { label: 'Quit', click: () => app.quit() },
];

contextMenu({ prepend: () => template });

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      nodeIntegration: true,
      contextIsolation: false,
      preload: path.join(__dirname, 'preload.js'),
    },
  });

  // Load the Vite application URL or build output
  const VITE_DEV_SERVER_URL = process.env.VITE_DEV_SERVER_URL;

  if (VITE_DEV_SERVER_URL) {
    mainWindow.loadURL(VITE_DEV_SERVER_URL);
  } else {
    mainWindow.loadFile(path.join(__dirname, 'index.html'));
  }

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

app.on('ready', createWindow);

ipcMain.handle('open-folder-dialog', async (event, arg) => {
  if (mainWindow) {
    const result = await dialog.showOpenDialog(mainWindow, {
      properties: ['openDirectory'],
    });
    return result;
  }
  return null;
});

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('activate', () => {
  if (BrowserWindow.getAllWindows().length === 0) {
    createWindow();
  }
});