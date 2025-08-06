import { NextRequest } from 'next/server';

export async function POST(req: NextRequest) {
  try {
    const { messages } = await req.json();
    const latestMessage = messages[messages.length - 1];

    const ollamaResponse = await fetch('http://localhost:11434/api/generate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model: 'tinyllama',
        prompt: latestMessage.content,
        stream: false,
      }),
    });

    if (!ollamaResponse.ok) {
      const errorText = await ollamaResponse.text();
      return new Response(`Error from Ollama: ${errorText}`, { status: ollamaResponse.status });
    }

    const data = await ollamaResponse.json();
    
    return new Response(JSON.stringify({ message: data.response }), {
      headers: { 'Content-Type': 'application/json' },
    });

  } catch (error) {
    console.error('Error in chat API:', error);
    return new Response('Internal Server Error', { status: 500 });
  }
}