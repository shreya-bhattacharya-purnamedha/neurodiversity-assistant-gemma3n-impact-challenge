export interface Message {
  id: string;
  content: string;
  role: 'user' | 'assistant';
  timestamp: Date;
}

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  error: string | null;
}

export interface QuickTopic {
  id: string;
  text: string;
  emoji: string;
  category: 'adhd' | 'autism' | 'anxiety' | 'depression' | 'general';
} 