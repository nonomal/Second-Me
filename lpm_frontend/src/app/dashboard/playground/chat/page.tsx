'use client';

import { useState, useEffect, useRef, useMemo } from 'react';
import ContextSettings from '@/components/playground/ContextSettings';
import ChatInput from '@/components/chat/ChatInput';
import ChatMessage from '@/components/chat/ChatMessage';
import ChatHistory from '@/components/chat/ChatHistory';
import {
  type ChatMessage as StorageMessage,
  type ChatSession,
  chatWithUploadStorage as chatStorage
} from '@/utils/chatStorage';
import type { ChatRequest } from '@/hooks/useSSE';
import { useSSE } from '@/hooks/useSSE';
import { useLoadInfoStore } from '@/store/useLoadInfoStore';
import { getTrainingParams } from '@/service/train';
import { getActiveCloudModel, getCurrentModelDisplayNameAsync } from '@/utils/cloudModelUtils';

// Use the Message type directly from storage
type Message = StorageMessage;

type ModelType = 'chat' | 'thinking';

interface PlaygroundSettings {
  enableL0Retrieval: boolean;
  enableL1Retrieval: boolean;
  enableHelperModel: boolean;
  selectedModel: string;
  apiKey: string;
  systemPrompt: string;
  temperature: number;
}

// Constants
const STORAGE_KEY_SETTINGS = 'playgroundSettings';

// Function to generate unique ID
const generateMessageId = () => {
  return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
};

export default function PlaygroundChat() {
  const loadInfo = useLoadInfoStore((state) => state.loadInfo);

  const { sendStreamMessage, streaming, streamContent, stopSSE } = useSSE();

  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Message[]>([]);
  const [modelType, setModelType] = useState<ModelType | undefined>(undefined);
  const [currentModelName, setCurrentModelName] = useState<string>('Loading...');

  const originPrompt = useMemo(() => {
    // const name = loadInfo?.name || 'user';

    return `Your mission is to help others with your background information and past memory by providing relevant answers and precise solutions.

    When thinking, please follow these steps in order and present the results clearly:
      1. Reflect on how the question relates to your background: Go through your memory and personal info, analyzing the connection between the question and your records.
      2. Derive the answer to the question: Use your historical data and the question's details to ensure the answer is accurate and relevant.
      3. Generate a high-quality response: Produce the best solution that meets the person's needs, presented systematically with high information density.`;
  }, [loadInfo, modelType]);
  const originSettings = useMemo(() => {
    return {
      enableL0Retrieval: false,
      enableL1Retrieval: false,
      enableHelperModel: false,
      selectedModel: 'ollama',
      apiKey: 'http://localhost:11434',
      systemPrompt: originPrompt,
      temperature: 0.3
    };
  }, [originPrompt]);

  const [settings, setSettings] = useState<PlaygroundSettings>(originSettings);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setSettings((prev) => {
      const newSettings = { ...prev, systemPrompt: originPrompt };

      localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(newSettings));

      return newSettings;
    });
  }, [originPrompt]);

  useEffect(() => {
    getTrainingParams()
      .then((res) => {
        if (res.data.code === 0) {
          const data = res.data.data;

          localStorage.setItem('trainingParams', JSON.stringify(data));
          setModelType(data.is_cot ? 'thinking' : 'chat');
        } else {
          throw new Error(res.data.message);
        }
      })
      .catch((error) => {
        console.error(error.message);
      });
  }, []);

  // Get current model display name
  useEffect(() => {
    const updateModelName = async () => {
      try {
        const modelName = await getCurrentModelDisplayNameAsync();
        setCurrentModelName(modelName);
      } catch (error) {
        console.error('Failed to get current model name:', error);
        setCurrentModelName('Model Status Unknown');
      }
    };

    updateModelName();

    // Update model name every 5 seconds to reflect service changes
    const interval = setInterval(updateModelName, 5000);

    return () => clearInterval(interval);
  }, []);

  const scrollToBottom = () => {
    if (messagesEndRef.current) {
      const container = messagesEndRef.current.parentElement;

      container?.scrollTo({
        top: container.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  const updateSessions = () => {
    const storedSessions = chatStorage.getSessions();

    setSessions(storedSessions);

    if (storedSessions.length == 0) {
      handleNewChat();
    }
  };

  // When messages are updated, scroll to the bottom
  useEffect(() => {
    scrollToBottom();
  }, [messages, streamContent]);

  // First initialization
  useEffect(() => {
    const storedSessions = chatStorage.getSessions();

    setSessions(storedSessions);

    if (storedSessions.length == 0) {
      handleNewChat();
    } else {
      setActiveSessionId(storedSessions[0].id);
      setMessages(storedSessions[0].messages);
    }
  }, []);

  // Load sessions, messages, and settings from storage on mount
  useEffect(() => {
    const storedSettings = localStorage.getItem(STORAGE_KEY_SETTINGS);

    if (storedSettings) {
      try {
        setSettings(JSON.parse(storedSettings));
      } catch (error) {
        console.error('Failed to parse stored settings:', error);
      }
    }
  }, []);

  const handleNewChat = () => {
    const newSession = chatStorage.createSession();

    setSessions((prev) => [newSession, ...prev]);
    setActiveSessionId(newSession.id);
    setMessages([]);
  };

  const handleSessionClick = (sessionId: string) => {
    setActiveSessionId(sessionId);
    stopSSE();
    const _sessions = chatStorage.getSessions();
    const session = _sessions.find((s) => s.id === sessionId);

    if (session) {
      setMessages(session.messages);
    }
  };

  const handleSettingsChange = (newSettings: PlaygroundSettings) => {
    setSettings(newSettings);
    localStorage.setItem(STORAGE_KEY_SETTINGS, JSON.stringify(newSettings));
  };

  const handleSendMessage = async (content: string) => {
    // Create user message
    const userMessage: Message = {
      id: generateMessageId(),
      content,
      role: 'user',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    };

    // Create an empty assistant message
    const assistantMessage: Message = {
      id: generateMessageId(),
      content: '',
      role: 'assistant',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    };

    // Update message list, adding user message and empty assistant message
    let newMessages = [...messages, userMessage];

    const systemMessage: Message = {
      id: generateMessageId(),
      content: originPrompt,
      role: 'system',
      timestamp: new Date().toLocaleTimeString([], {
        hour: '2-digit',
        minute: '2-digit'
      })
    };

    if (!newMessages.find((item) => item.role === 'system')) {
      newMessages = [systemMessage, ...newMessages];
    } else {
      newMessages = newMessages.map((msg) => {
        if (msg.role === 'system') {
          return { ...msg, content: originPrompt };
        }

        return msg;
      });
    }

    setMessages([...newMessages, assistantMessage]);

    // Save messages to session
    if (activeSessionId) {
      chatStorage.saveSessionMessages(activeSessionId, [...newMessages, assistantMessage]);

      // If it's the first message in a new session, update session title
      if (messages.length === 0) {
        const title = content.length > 30 ? content.substring(0, 30) + '...' : content;

        chatStorage.updateSession(activeSessionId, { title, lastMessage: content });
        setSessions(chatStorage.getSessions());
      }
    }

    // Send request
    const chatRequest: ChatRequest = {
      messages: newMessages.map((msg) => ({
        role: msg.role,
        content: msg.content
      })),
      metadata: {
        enable_l0_retrieval: settings.enableL0Retrieval,
        enable_l1_retrieval: settings.enableL1Retrieval
      },
      temperature: settings.temperature,
      stream: true
    };

    // Check if a cloud model is active
    const cloudModel = getActiveCloudModel();

    if (cloudModel) {
      await sendStreamMessage(chatRequest, true, cloudModel.deployed_model);
    } else {
      await sendStreamMessage(chatRequest);
    }
  };

  // Listen for streamContent changes to update messages
  useEffect(() => {
    if (!streamContent) return;

    const newMessages = messages.map((msg, index) => {
      if (index === messages.length - 1 && msg.role === 'assistant') {
        return { ...msg, content: streamContent };
      }

      return msg;
    });

    setMessages(newMessages);

    if (activeSessionId) {
      chatStorage.updateSession(activeSessionId, {
        lastMessage: streamContent,
        messages: newMessages
      });
    }
  }, [streamContent]);

  const handleClearChat = () => {
    if (activeSessionId) {
      chatStorage.saveSessionMessages(activeSessionId, []);
      setMessages([]);
    }

    updateSessions();
  };

  const handleDeleteChat = (sessionId: string) => {
    chatStorage.deleteSession(sessionId);
    updateSessions();

    if (sessionId === activeSessionId) {
      const storedSessions = chatStorage.getSessions();

      setActiveSessionId(storedSessions[0]?.id);
      setMessages(storedSessions[0]?.messages);
    }
  };

  return (
    <div className="h-full w-full flex">
      <ChatHistory
        activeSessionId={activeSessionId}
        onDeleteChat={handleDeleteChat}
        onNewChat={handleNewChat}
        onSessionClick={handleSessionClick}
        sessions={sessions}
      />
      {/* Main chat area */}
      <div className="flex-1 flex flex-col bg-white">
        <div className="flex items-center justify-between px-6 py-3 border-b">
          <div className="flex items-center gap-3">
            <h2 className="text-lg font-semibold">Chat with Second Me</h2>
            <span className="text-sm text-gray-500 bg-gray-100 px-2 py-1 rounded">
              {currentModelName}
            </span>
          </div>
          <button className="text-sm text-gray-600 hover:text-gray-900" onClick={handleClearChat}>
            Clear Chat
          </button>
        </div>
        {/* Chat message area */}
        <div className="flex-1 overflow-y-auto px-6 py-6">
          {messages.length === 0 ? (
            <div className="h-full flex items-center justify-center text-gray-400">
              Start a new conversation...
            </div>
          ) : (
            <>
              <div className="mx-auto w-full space-y-6">
                {messages.map((message, index) => (
                  <ChatMessage
                    key={message.id}
                    isLoading={
                      streaming &&
                      !streamContent &&
                      index === messages.length - 1 &&
                      message.role === 'assistant'
                    }
                    message={message.content}
                    role={message.role}
                    timestamp={message.timestamp}
                  />
                ))}
              </div>
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Input area */}
        <div className="flex-shrink-0 border-t border-gray-200 p-4">
          <div className="mx-auto">
            <ChatInput disabled={streaming} onSendMessage={handleSendMessage} />
          </div>
        </div>
      </div>

      {/* Right settings panel */}
      <div className="w-80 border-l border-gray-200 bg-white">
        <ContextSettings onSettingsChange={handleSettingsChange} settings={settings} />
      </div>
    </div>
  );
}
