export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface QuickTopic {
  id: string;
  text: string;
  emoji: string;
  category: 'adhd' | 'autism' | 'anxiety' | 'general';
}
