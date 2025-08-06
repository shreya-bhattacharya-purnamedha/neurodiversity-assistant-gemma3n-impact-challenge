'use client';

import { useState, useRef, useEffect } from 'react';
import { Message, QuickTopic } from '@/lib/types';
import ClientOnly from './components/ClientOnly';

const quickTopics: QuickTopic[] = [
  { id: '1', text: 'ADHD focus strategies', emoji: '🎯', category: 'adhd' },
  { id: '2', text: 'Sensory processing help', emoji: '🎨', category: 'autism' },
  { id: '3', text: 'Anxiety management', emoji: '🧘‍♀️', category: 'anxiety' },
  { id: '4', text: 'Social communication', emoji: '💬', category: 'autism' },
  { id: '5', text: 'Executive functioning', emoji: '🧠', category: 'adhd' },
  { id: '6', text: 'Emotional regulation', emoji: '⚖️', category: 'anxiety' },
  { id: '7', text: 'Workplace accommodations', emoji: '💼', category: 'general' },
  { id: '8', text: 'Self-advocacy skills', emoji: '🗣️', category: 'general' }
];

const renderMarkdown = (text: string, isUserMessage: boolean = false) => {
  if (!text) return '';
  
  const textColor = isUserMessage ? 'text-white' : 'text-gray-900';
  const headerColor = isUserMessage ? 'text-white' : 'text-gray-900';
  const borderColor = isUserMessage ? 'border-white' : 'border-gray-200';
  
  return text
    .replace(/\*\*(.*?)\*\*/g, `<strong class="font-semibold ${textColor}">$1</strong>`)
    .replace(/\*(.*?)\*/g, `<em class="italic ${textColor}">$1</em>`)
    .replace(/^## (.*$)/gim, `<h2 class="font-bold text-xl mb-3 border-b ${borderColor} pb-2 ${headerColor}">$1</h2>`)
    .replace(/^### (.*$)/gim, `<h3 class="font-bold text-lg mb-2 ${headerColor}">$1</h3>`)
    .replace(/^[-*] (.*$)/gim, `<li class="ml-6 mb-1 ${textColor}">• $1</li>`)
    .replace(/^\d+\. (.*$)/gim, `<li class="ml-6 mb-1 ${textColor}">$&</li>`)
    .replace(/\n\n/g, `</p><p class="mb-3 leading-relaxed ${textColor}">`)
    .replace(/\n/g, '<br>')
    .replace(/^(.*)$/, `<p class="mb-3 leading-relaxed ${textColor}">$1</p>`)
    .replace(/<p class="mb-3 leading-relaxed [^\"]*">\s*<\/p>/g, '')
    .replace(/<p class="mb-3 leading-relaxed [^\"]*">\s*<br>\s*<\/p>/g, '');
};

export default function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      content: input,
      role: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: [...messages, { role: 'user', content: input }]
        }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: data.message,
        role: 'assistant',
        timestamp: new Date()
      };

      setMessages(prev => [...prev, assistantMessage]);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: 'I\'m having trouble connecting right now. Please try again in a moment.',
        role: 'assistant',
        timestamp: new Date()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const clearChat = () => setMessages([]);

  return (
    <ClientOnly
      fallback={
        <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
          <div className="flex items-center justify-center h-full">
            <div className="text-center">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white text-2xl">🧠</span>
              </div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">Loading...</h2>
            </div>
          </div>
        </div>
      }
    >
      <div className="flex flex-col h-screen bg-gradient-to-br from-blue-50 to-indigo-100">
        {/* Header */}
        <div className="bg-white shadow-sm border-b border-gray-200 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center">
                <span className="text-white text-lg font-bold">🧠</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Neurodiversity Assistant</h1>
                <p className="text-sm text-gray-600">Evidence-based support for neurodiverse individuals</p>
              </div>
            </div>
            <button
              onClick={clearChat}
              className="px-3 py-1 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-md transition-colors"
            >
              Clear Chat
            </button>
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-6 py-4 space-y-4">
          {messages.length === 0 && (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-white text-2xl">🧠</span>
              </div>
              <h2 className="text-xl font-semibold text-gray-900 mb-2">Welcome to Neurodiversity Support</h2>
              <p className="text-gray-600 mb-6 max-w-md mx-auto">
                I'm here to provide evidence-based information and practical strategies for neurodiverse individuals. 
                Start a conversation or choose a topic below.
              </p>
              
              {/* Quick Topics */}
              <div className="grid grid-cols-2 gap-3 max-w-lg mx-auto">
                {quickTopics.map((topic) => (
                  <button
                    key={topic.id}
                    onClick={() => setInput(topic.text)}
                    className="p-3 bg-white rounded-lg border border-gray-200 hover:border-blue-300 hover:shadow-md transition-all text-left"
                  >
                    <div className="text-lg mb-1">{topic.emoji}</div>
                    <div className="text-sm font-medium text-gray-900">{topic.text}</div>
                  </button>
                ))}
              </div>
            </div>
          )}

          {messages.map((message) => (
            <div
              key={message.id}
              className={`flex ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>
              <div
                className={`max-w-3xl px-4 py-3 rounded-lg ${ 
                  message.role === 'user'
                    ? 'bg-blue-500 text-white'
                    : 'bg-white text-gray-900 shadow-sm border border-gray-200'
                }`}>
                <div
                  className="max-w-none"
                  dangerouslySetInnerHTML={{
                    __html: renderMarkdown(message.content, message.role === 'user')
                  }}
                  suppressHydrationWarning
                />
                <div className={`text-xs mt-2 ${ 
                  message.role === 'user' ? 'text-blue-100' : 'text-gray-500'
                }`}>
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-white text-gray-900 shadow-sm border border-gray-200 px-4 py-3 rounded-lg">
                <div className="flex items-center space-x-2">
                  <div className="flex space-x-1">
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                    <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  </div>
                  <span className="text-sm text-gray-600">Thinking...</span>
                </div>
              </div>
            </div>
          )}
          
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="bg-white border-t border-gray-200 px-6 py-4">
          <form onSubmit={handleSendMessage} className="flex space-x-3">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask me about neurodiversity, coping strategies, or anything else..."
              className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent text-gray-900"
              disabled={isLoading}
            />
            <button
              type="submit"
              disabled={!input.trim() || isLoading}
              className="px-6 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              Send
            </button>
          </form>
        </div>
      </div>
    </ClientOnly>
  );
}
