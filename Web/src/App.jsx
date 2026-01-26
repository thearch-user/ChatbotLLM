import { useState } from 'react'

export default function App() {
const [messages, setMessages] = useState([
{ role: 'assistant', content: 'Hi! I am your local LLM.' }
])
const [input, setInput] = useState('')


async function send() {
if (!input.trim()) return


const userMsg = { role: 'user', content: input }
setMessages(m => [...m, userMsg])
setInput('')


const res = await fetch('/chat', {
method: 'POST',
headers: { 'Content-Type': 'application/json' },
body: JSON.stringify({ message: input })
})


const data = await res.json()
setMessages(m => [...m, { role: 'assistant', content: data.reply }])
}


return (
<div className="h-screen flex flex-col text-zinc-100">


{/* Header */}
<header className="h-14 border-b border-white/10 flex items-center px-6 bg-zinc-900">
<h1 className="font-semibold">Local LLM</h1>
</header>


{/* Chat */}
<main className="flex-1 overflow-y-auto px-6 py-8 space-y-6">
{messages.map((m, i) => (
<div key={i} className={m.role === 'user' ? 'flex justify-end' : 'flex justify-start'}>
<div className={
m.role === 'user'
? 'bg-indigo-600 px-5 py-3 rounded-2xl max-w-[70%]'
: 'bg-zinc-800 px-5 py-3 rounded-2xl max-w-[70%]'
}>
{m.content}
</div>
</div>
))}
</main>


{/* Input */}
<footer className="border-t border-white/10 p-4 bg-zinc-900">
<div className="flex gap-3">
<textarea
value={input}
onChange={e => setInput(e.target.value)}
onKeyDown={e => e.key === 'Enter' && !e.shiftKey && (e.preventDefault(), send())}
placeholder="Send a message"
className="flex-1 resize-none rounded-xl bg-zinc-800 px-4 py-3 outline-none focus:ring-2 focus:ring-indigo-500"
/>
<button
onClick={send}
className="px-5 rounded-xl bg-indigo-600 hover:bg-indigo-500 font-medium"
>
Send
</button>
</div>
</footer>


</div>
  )
}