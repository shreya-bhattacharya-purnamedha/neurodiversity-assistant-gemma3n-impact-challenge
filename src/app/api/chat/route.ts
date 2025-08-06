import { NextRequest, NextResponse } from 'next/server';
import { SYSTEM_PROMPT } from '@/lib/prompt';

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

const OLLAMA_URL = 'http://localhost:11434/api/generate';
const TIMEOUT_MS = 30000;

export async function POST(req: NextRequest) {
  try {
    const { messages }: { messages: ChatMessage[] } = await req.json();

    if (!messages?.length) {
      return NextResponse.json(
        { error: 'Invalid messages format' },
        { status: 400 }
      );
    }

    const userMessage = messages[messages.length - 1]?.content || '';
    const response = await handleOllamaRequest(userMessage);

    return NextResponse.json({ message: response });
  } catch (error) {
    console.error('API Error:', error);
    return NextResponse.json(
      { error: 'Internal Server Error' },
      { status: 500 }
    );
  }
}

async function handleOllamaRequest(userMessage: string): Promise<string> {
  const prompt = `${SYSTEM_PROMPT}\n\nUser: ${userMessage}\nAssistant:`;

  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), TIMEOUT_MS);
    
    const response = await fetch(OLLAMA_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        model: 'gemma3n:e2b',
        prompt,
        stream: false,
        options: {
          temperature: 0.7,
          top_p: 0.9,
          num_predict: 4096,
        },
      }),
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      throw new Error(`Ollama API error: ${response.status}`);
    }

    const data = await response.json();
    
    if (data.error) {
      throw new Error(`Ollama error: ${data.error}`);
    }
    
    return data.response || 'I apologize, but I am unable to respond at the moment.';
  } catch (error) {
    console.error('Ollama request failed:', error);
    return getFallbackResponse(userMessage);
  }
}

function getFallbackResponse(userMessage: string): string {
  const message = userMessage.toLowerCase();
  
  if (message.includes('anxiety') || message.includes('stress') || message.includes('worried')) {
    return `🧘‍♀️ **Anxiety & Stress Management**

I understand you're dealing with anxiety and stress. Here are some **evidence-based techniques** that can help:

**🌿 Immediate Relief Techniques:**
1. **Deep Breathing (4-7-8 Method)**
   - Inhale for 4 counts, hold for 7, exhale for 8
   - Repeat 4 times

2. **5-4-3-2-1 Grounding Exercise**
   - Name 5 things you can see, 4 you can touch, 3 you can hear, 2 you can smell, 1 you can taste

**🎯 Long-term Strategies:**
- **Regular Exercise**: 30 minutes daily can reduce anxiety by 30%
- **Mindfulness Meditation**: 10 minutes daily improves stress response
- **Sleep Hygiene**: 7-9 hours of quality sleep
- **Social Connection**: Talk to trusted friends or family

**💡 Professional Support:**
If anxiety persists, consider speaking with a mental health professional.

**What specific aspect of anxiety would you like to explore further?** I'm here to support you! 💙`;
  }
  
  if (message.includes('adhd') || message.includes('focus') || message.includes('attention')) {
    return `🎯 **ADHD & Focus Support**

I understand you're looking for ADHD and focus support. Here are **research-backed strategies**:

**⚡ Immediate Focus Techniques:**
1. **Pomodoro Method**: Work for 25 minutes, take 5-minute breaks
2. **Body Doubles**: Work alongside someone else (increases productivity by 40%)
3. **Environmental Modifications**: Remove distractions, use noise-canceling headphones

**🧠 Executive Function Support:**
- **Visual Timers**: Use apps like Forest or Focus@Will
- **Task Breakdown**: Break large tasks into smaller steps
- **External Reminders**: Use phone alarms and sticky notes
- **Routine Building**: Create consistent daily schedules

**💊 Medical Support:**
ADHD may benefit from medication (when prescribed) and behavioral therapy.

**What specific challenge would you like to work on?** I'm here to help! 🌟`;
  }
  
  if (message.includes('autism') || message.includes('sensory') || message.includes('social')) {
    return `🌈 **Autism & Sensory Support**

I understand you're looking for autism and sensory support. Here are **evidence-based strategies**:

**🎧 Sensory Regulation Techniques:**
1. **Sensory Diet**: Heavy work activities, deep pressure, oral sensory input
2. **Environmental Modifications**: Reduce fluorescent lighting, use noise-canceling headphones

**🤝 Social Communication Support:**
- **Visual Schedules**: Use pictures and timers
- **Social Stories**: Prepare for new situations
- **Scripts**: Practice common conversations

**🏠 Daily Living Skills:**
- **Routine Building**: Create predictable schedules
- **Task Analysis**: Break down complex activities
- **Visual Supports**: Use pictures and checklists

**What specific area would you like to explore?** Every person is unique! 🌟`;
  }
  
  if (message.includes('depression') || message.includes('mood') || message.includes('sad')) {
    return `💙 **Depression & Mood Support**

I understand you're dealing with depression and mood challenges. Here are **evidence-based strategies**:

**🌅 Daily Wellness Practices:**
1. **Morning Routine**: Start with 5 minutes of gentle movement, open curtains
2. **Social Connection**: Reach out to one person daily, join support groups

**🎯 Behavioral Activation:**
- **Small Steps**: Start with 5-minute activities
- **Pleasant Activities**: Do things you used to enjoy
- **Achievement Activities**: Complete small tasks for accomplishment

**💊 Professional Support:**
Depression often requires therapy, medication (when prescribed), and lifestyle changes.

**🚨 Crisis Support:**
If you're having thoughts of self-harm, please reach out immediately:
- National Suicide Prevention Lifeline: 988
- Crisis Text Line: Text HOME to 741741

**What would be most helpful for you right now?** I'm here to support you. 💙`;
  }
  
  return `💙 **Welcome to Neurodiversity Support!**

I'm here to provide **evidence-based information and support** for neurodiverse individuals. I can help you with:

**🧠 Common Topics:**
- **ADHD**: Focus strategies, executive function support
- **Autism**: Sensory regulation, social communication
- **Anxiety & Depression**: Coping strategies, stress management
- **Learning Differences**: Study strategies, accommodations

**🎯 What I Offer:**
- **Research-backed techniques** validated by professionals
- **Practical strategies** you can implement immediately
- **Supportive guidance** without judgment

**What would you like to explore today?** 🌟`;
} 