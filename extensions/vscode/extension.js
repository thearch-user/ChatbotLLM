const vscode = require('vscode');
const http = require('http');

function activate(context) {
    let disposable = vscode.commands.registerCommand('chatbotllm.complete', async function () {
        const editor = vscode.window.activeTextEditor;
        if (!editor) return;

        const selection = editor.selection;
        const text = editor.document.getText(selection) || editor.document.getText();

        vscode.window.withProgress({
            location: vscode.ProgressLocation.Notification,
            title: "ChatbotLLM is thinking...",
            cancellable: false
        }, async (progress) => {
            try {
                const response = await callBackend(text);
                editor.edit(editBuilder => {
                    editBuilder.insert(selection.end, "\n" + response.prediction);
                });
            } catch (err) {
                vscode.window.showErrorMessage("ChatbotLLM Error: " + err.message);
            }
        });
    });

    context.subscriptions.push(disposable);
}

function callBackend(text) {
    return new Promise((resolve, reject) => {
        const data = JSON.stringify({ text: text });
        const options = {
            hostname: 'localhost',
            port: 8000,
            path: '/predict',
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Content-Length': data.length
            }
        };

        const req = http.request(options, (res) => {
            let body = '';
            res.on('data', (chunk) => body += chunk);
            res.on('end', () => {
                if (res.statusCode === 200) {
                    resolve(JSON.parse(body));
                } else {
                    reject(new Error("Status " + res.statusCode));
                }
            });
        });

        req.on('error', (e) => reject(e));
        req.write(data);
        req.end();
    });
}

function deactivate() {}

module.exports = {
    activate,
    deactivate
};
