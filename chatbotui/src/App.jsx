import React, { useState, useRef, useEffect } from 'react';
import { ArrowLeft, MoreHorizontal, ThumbsUp, ThumbsDown, Copy, Send, LayoutGrid, Bot, MessageCircle, X, Minus } from 'lucide-react';
import Robot3D from './components/Robot3D';
import MiniRobot from './components/MiniRobot';
import WaitingRobot from './components/WaitingRobot';

const INITIAL_MESSAGES = [
  {
    id: 1,
    role: 'bot',
    content: 'Hello! I am the Prime Source Global assistant. How can I help you today?',
  }
];

function App() {
  const [messages, setMessages] = useState(INITIAL_MESSAGES);
  const [inputValue, setInputValue] = useState('');
  const [isOpen, setIsOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleSend = async () => {
    if (!inputValue.trim()) return;
    
    const userMsg = {
      id: Date.now(),
      role: 'user',
      content: inputValue,
    };
    
    setMessages(prev => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const response = await fetch('http://localhost:8000/chat', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: inputValue }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      
      const botMsg = {
        id: Date.now() + 1,
        role: 'bot',
        content: data.answer,
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (error) {
      console.error('Error:', error);
      const errorMsg = {
        id: Date.now() + 1,
        role: 'bot',
        content: "Sorry, I encountered an error while connecting to the server.",
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="relative min-h-screen bg-zinc-950 font-sans text-zinc-100 selection:bg-cyan-500/30 flex items-center justify-center">
      {/* Background Content (Demo Page) */}
      <div className="p-10 text-center">
        <div className="animate-fade-in-up-smooth">
          <div className="mb-6 inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-zinc-900 border border-zinc-800 shadow-2xl shadow-cyan-500/10 mb-8 animate-float-more">
             <Bot className="w-10 h-10 text-[#22d3ee]" />
          </div>
        </div>
        <h1 className="text-5xl font-bold mb-6 text-white tracking-tight animate-fade-in-up-smooth animation-delay-200">
          Welcome to <span className="text-transparent bg-clip-text bg-gradient-to-r from-[#22d3ee] via-cyan-400 to-blue-500 animate-text-shimmer">AI Service</span>
        </h1>
        <p className="text-xl text-zinc-400 max-w-2xl mx-auto leading-relaxed animate-fade-in-up-smooth animation-delay-400">
          Experience the future of conversation. Click the chat icon in the bottom right to start your journey with our advanced AI assistant.
        </p>
      </div>

      {/* Floating Chat Widget */}
      <div className={`fixed bottom-6 right-6 z-50 transition-all duration-500 ease-[cubic-bezier(0.16,1,0.3,1)] transform origin-bottom-right ${isOpen ? 'scale-100 opacity-100 translate-y-0' : 'scale-90 opacity-0 translate-y-8 pointer-events-none'}`}>
        <div className="w-[380px] h-[600px] bg-[#09090b] text-white rounded-2xl shadow-2xl flex flex-col overflow-hidden border border-zinc-800 ring-1 ring-black/5">
          
          {/* Widget Header */}
          <div className="flex items-center justify-between px-4 py-3 bg-[#09090b] border-b border-zinc-800">
            <div className="flex items-center gap-3">
               <div className="scale-75 origin-left">
                  <MiniRobot />
               </div>
               <div>
                  <h3 className="text-[15px] font-semibold">Chat with AI Bot</h3>
                  <div className="flex items-center gap-1.5">
                     <span className="w-1.5 h-1.5 bg-green-500 rounded-full"></span>
                     <span className="text-[11px] text-zinc-400">Online</span>
                  </div>
               </div>
            </div>
            <div className="flex items-center gap-1">
               <button onClick={() => setIsOpen(false)} className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition">
                  <Minus className="w-5 h-5" />
               </button>
               <button onClick={() => setIsOpen(false)} className="p-2 hover:bg-zinc-800 rounded-full text-zinc-400 hover:text-white transition">
                  <X className="w-5 h-5" />
               </button>
            </div>
          </div>

          {/* Chat Area */}
          <div className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin scrollbar-thumb-zinc-700 scrollbar-track-transparent perspective-[1000px]">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === 'user' ? 'justify-end animate-user-message-in' : 'justify-start animate-message-in'}`}
              >
                {msg.role === 'user' ? (
                  <div className="bg-[#22d3ee] text-black px-4 py-2.5 rounded-2xl rounded-tr-sm max-w-[85%] text-[14px] font-medium leading-relaxed shadow-md">
                    {msg.content}
                  </div>
                ) : (
                  <div className="flex gap-3 max-w-[90%] animate-fade-in">
                     <div className="w-10 h-10 flex-shrink-0 -ml-1">
                        <div className="scale-[0.6] origin-top-left">
                            <MiniRobot />
                        </div>
                     </div>
                     <div className="flex flex-col gap-2">
                        <div className="bg-[#1c1c1e] p-3 rounded-2xl rounded-tl-sm border border-zinc-800/50 text-zinc-200 text-[14px] leading-6 font-normal">
                          {msg.content}
                        </div>
                        {/* Bot Actions */}
                        <div className="flex items-center gap-1">
                          <button className="p-1.5 rounded-full hover:bg-zinc-800 transition text-zinc-500 hover:text-white">
                            <ThumbsUp className="w-3.5 h-3.5" />
                          </button>
                          <button className="p-1.5 rounded-full hover:bg-zinc-800 transition text-zinc-500 hover:text-white">
                            <ThumbsDown className="w-3.5 h-3.5" />
                          </button>
                          <button className="p-1.5 rounded-full hover:bg-zinc-800 transition text-zinc-500 hover:text-white">
                            <Copy className="w-3.5 h-3.5" />
                          </button>
                        </div>
                     </div>
                  </div>
                )}
              </div>
            ))}
            
            {isLoading && (
              <div className="flex justify-start animate-message-in">
                <div className="flex gap-3 max-w-[90%] animate-fade-in">
                   <div className="flex flex-col gap-2">
                      <div className="bg-[#1c1c1e] p-4 rounded-2xl rounded-tl-sm border border-zinc-800/50 flex items-center gap-4">
                        <div className="scale-75 origin-left">
                           <WaitingRobot />
                        </div>
                        <span className="text-zinc-400 text-sm animate-pulse">Request from PSG is coming...</span>
                      </div>
                   </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input Area */}
          <div className="p-4 bg-[#09090b] border-t border-zinc-800">
            <div className="flex items-center gap-2 bg-[#1c1c1e] p-1.5 pl-3 pr-1.5 rounded-[24px] border border-zinc-800/50 focus-within:border-zinc-700 transition duration-300">
               <button className="p-1.5 hover:bg-zinc-800 rounded-full text-zinc-400 transition">
                  <LayoutGrid className="w-5 h-5" strokeWidth={1.5} />
               </button>
               
               <input 
                  type="text" 
                  placeholder="Type message..." 
                  className="flex-1 bg-transparent text-white placeholder-zinc-500 outline-none text-[14px]"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSend()}
               />
               
               <button 
                  onClick={handleSend}
                  disabled={!inputValue.trim()}
                  className={`p-2 rounded-full transition-all duration-300 ${inputValue.trim() ? 'bg-[#22d3ee] text-black rotate-0' : 'bg-transparent text-zinc-600 rotate-90'}`}
               >
                  <Send className="w-4 h-4" fill={inputValue.trim() ? "currentColor" : "none"} />
               </button>
            </div>
            <div className="text-center mt-2 text-[10px] text-zinc-500">
               AI can make mistakes.
            </div>
          </div>
        </div>
      </div>

      {/* Floating Toggle Button / 3D Robot */}
      <div 
        className={`fixed bottom-6 right-6 transition-all duration-300 z-40 ${isOpen ? 'scale-0 opacity-0 pointer-events-none' : 'scale-100 opacity-100'}`}
      >
        <Robot3D onClick={() => setIsOpen(true)} notificationCount={1} />
      </div>

      {/* Close Button when open (Optional alternative to header close) */}
       <button 
        onClick={() => setIsOpen(false)}
        className={`fixed bottom-6 right-6 p-4 rounded-full shadow-2xl transition-all duration-300 hover:scale-105 z-40 bg-zinc-800 text-white ${isOpen ? 'scale-100 opacity-100' : 'scale-0 opacity-0 pointer-events-none'}`}
      >
        <X className="w-7 h-7" />
      </button>

    </div>
  );
}

export default App;
