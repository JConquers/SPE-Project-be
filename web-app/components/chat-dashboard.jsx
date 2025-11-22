'use client'

import { useState, useEffect, useRef } from 'react'
import { Send, Menu, Home, MessageSquare, Settings } from 'lucide-react';
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar"

export function ChatDashboardComponent() {
  const questions = [
    "What personal values guide your decision-making in challenging business situations?",
    "How do you maintain a balance between ambition and realism when setting business goals?",
    "How do you perceive setbacks or failures in your business journey?",
    "What mindset helps you persevere during periods of uncertainty?",
    "How do you manage self-doubt or criticism in the context of running your business?",
    "What strategies do you use to stay motivated and inspire your team?",
    "How do you approach learning from competitors without losing sight of your own goals?",
    "What role does trust play in your relationships with employees and stakeholders?",
    "How do you manage the emotional impact of financial pressure in your business?",
    "How does your personal vision align with the overall goals of your business?",
    "Describe how you see you business future in the next 5 years.",
  ];
  

  const [currentQuestionIndex, setCurrentQuestionIndex] = useState(0);
  const [messages, setMessages] = useState([
    { id: 1, sender: 'ai', content: questions[0] }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [answers, setAnswers] = useState({});
  const scrollAreaRef = useRef(null)

  const [shapes, setShapes] = useState([])

  useEffect(() => {
    const generateShapes = () => {
      const shapeTypes = [
        { type: 'circle', count: 5 },
        { type: 'hexagon', count: 3 },
        { type: 'triangle', count: 4 }
      ]

      const newShapes = shapeTypes.flatMap(({ type, count }) =>
        Array.from({ length: count }, (_, i) => ({
          type,
          x: 10 + (i * 20) + (Math.random() * 10 - 5),
          y: 10 + (i * 20) + (Math.random() * 10 - 5),
          size: 80 + Math.random() * 40,
          opacity: 0.1 + Math.random() * 0.05,
          rotation: Math.random() * 360,
        })))

      setShapes(newShapes)
    }

    generateShapes()
  }, [])

  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollTop = scrollAreaRef.current.scrollHeight
    }
  }, [messages])

  const renderShape = (shape) => {
    const lightPurple = '#e0d0ff'
    switch (shape.type) {
      case 'circle':
        return (
          (<circle
            cx={`${shape.x}%`}
            cy={`${shape.y}%`}
            r={shape.size / 2}
            fill={lightPurple}
            opacity={shape.opacity} />)
        );
      case 'hexagon':
        return (
          (<polygon
            points={`${shape.x},${shape.y} ${shape.x + shape.size / 2},${shape.y + shape.size * 0.29} ${shape.x + shape.size / 2},${shape.y + shape.size * 0.71} ${shape.x},${shape.y + shape.size} ${shape.x - shape.size / 2},${shape.y + shape.size * 0.71} ${shape.x - shape.size / 2},${shape.y + shape.size * 0.29}`}
            fill={lightPurple}
            opacity={shape.opacity}
            transform={`rotate(${shape.rotation} ${shape.x} ${shape.y})`} />)
        );
      case 'triangle':
        return (
          (<polygon
            points={`${shape.x},${shape.y} ${shape.x + shape.size / 2},${shape.y + shape.size * 0.87} ${shape.x - shape.size / 2},${shape.y + shape.size * 0.87}`}
            fill={lightPurple}
            opacity={shape.opacity}
            transform={`rotate(${shape.rotation} ${shape.x} ${shape.y})`} />)
        );
      default:
        return null
    }
  }



  const handleSendMessage = () => {
    if (inputMessage.trim() === '') return;

    const newUserMessage = {
      id: messages.length + 1,
      sender: 'user',
      content: inputMessage.trim()
    };

    setMessages((prevMessages) => [...prevMessages, newUserMessage]);
    setAnswers((prevAnswers) => ({
      ...prevAnswers,
      [questions[currentQuestionIndex]]: inputMessage.trim()
    }));
    setInputMessage('');

    // Simulate AI response and move to the next question
    setTimeout(() => {
      if (currentQuestionIndex < questions.length - 1) {
        const nextQuestion = questions[currentQuestionIndex + 1];
        const aiResponse = {
          id: messages.length + 2,
          sender: 'ai',
          content: nextQuestion
        };

        setMessages((prevMessages) => [...prevMessages, aiResponse]);
        setCurrentQuestionIndex((prevIndex) => prevIndex + 1);
      } else {
        const finalResponse = {
          id: messages.length + 2,
          sender: 'ai',
          content: "Thank you for completing all the questions!"
        };

        setMessages((prevMessages) => [...prevMessages, finalResponse]);
      }
    }, 1000);
  };

  return (
    (<div className="flex h-screen bg-[#0a0b1a] overflow-hidden relative">
      {/* Background Shapes */}
      <svg
        className="absolute inset-0 w-full h-full"
        xmlns="http://www.w3.org/2000/svg">
        {shapes.map((shape, index) => (
          <g key={index}>{renderShape(shape)}</g>
        ))}
      </svg>
      {/* Sidebar */}
      <div
        className="w-64 bg-[#12132d]/80 backdrop-blur-sm p-4 flex flex-col border-r border-[#2a2b4a] z-10">
        <div className="flex items-center gap-2 mb-8">
          <div className="text-violet-400 font-bold text-2xl">Nano AI Chat</div>
        </div>
        
        <nav className="space-y-2">
          <Button
            variant="ghost"
            className="w-full justify-start gap-2 text-gray-300 hover:text-white hover:bg-[#1a1b3d]">
            <Home className="h-4 w-4" />
            Overview
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start gap-2 text-gray-300 hover:text-white hover:bg-[#1a1b3d]">
            <MessageSquare className="h-4 w-4" />
            Chats
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start gap-2 text-gray-300 hover:text-white hover:bg-[#1a1b3d]">
            <Settings className="h-4 w-4" />
            Settings
          </Button>
        </nav>
      </div>
      {/* Main Content */}
      <div className="flex-1 flex flex-col z-10">
        {/* Header */}
        <header
          className="h-16 border-b border-[#2a2b4a] flex items-center justify-between px-6 bg-[#12132d]/80 backdrop-blur-sm">
          <div className="flex items-center gap-4">
            <Menu className="h-6 w-6 text-gray-400" />
            <h1 className="text-lg font-semibold text-white">Chat Assistant</h1>
          </div>
          <div className="flex items-center gap-4">
            <Input
              placeholder="Search messages..."
              className="w-64 bg-[#1a1b3d] border-[#2a2b4a] text-white" />
            <Avatar>
              <AvatarImage src="/placeholder.svg" alt="User avatar" />
              <AvatarFallback>JD</AvatarFallback>
            </Avatar>
          </div>
        </header>

        {/* Chat Area */}
        <ScrollArea className="flex-1 p-4" ref={scrollAreaRef}>
          <div className="space-y-4">
            {messages.map((message) => (
              <div
                key={message.id}
                className={`flex ${message.sender === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div
                  className={`max-w-[80%] rounded-lg p-4 ${
                    message.sender === 'user'
                      ? 'bg-violet-600 text-white'
                      : 'bg-[#1a1b3d]/80 backdrop-blur-sm text-gray-100'
                  }`}>
                  {message.content}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {/* Input Area */}
        <div
          className="p-4 border-t border-[#2a2b4a] bg-[#12132d]/80 backdrop-blur-sm">
          <div className="flex gap-4">
            <Input
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={(e) => e.key === 'Enter' && handleSendMessage()}
              placeholder="Type your message..."
              className="flex-1 bg-[#1a1b3d] border-[#2a2b4a] text-white" />
            <Button onClick={handleSendMessage} className="bg-violet-600 hover:bg-violet-700">
              <Send className="h-4 w-4" />
              <span className="sr-only">Send message</span>
            </Button>
          </div>
        </div>
      </div>
    </div>)
  );
}