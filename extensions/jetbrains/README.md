# JetBrains Integration for ChatbotLLM

Currently, the easiest way to use ChatbotLLM in JetBrains (IntelliJ, PyCharm, WebStorm) is via the **Restful Tool** or by configuring a custom completion provider.

## Option 1: Using "HTTP Client" (Built-in)

1. Create a `chatbot.http` file in your project:
```http
POST http://localhost:8000/predict
Content-Type: application/json

{
  "text": "Write a python function for binary search"
}
```
2. Click the green "Run" icon in the gutter to see the prediction.

## Option 2: Custom Plugin Development
To build a dedicated plugin, you would use the [IntelliJ Platform SDK](https://plugins.jetbrains.com/docs/intellij/welcome.html). 

A basic `plugin.xml` would contribute an `action`:
```xml
<actions>
  <action id="ChatbotLLM.Complete" class="com.chatbotllm.CompleteAction" text="ChatbotLLM Complete">
    <add-to-group group-id="EditorPopupMenu" anchor="last"/>
    <keyboard-shortcut keymap="$default" first-keystroke="control alt C"/>
  </action>
</actions>
```

The `CompleteAction.kt` would use `java.net.http.HttpClient` to call `localhost:8000/predict`.
