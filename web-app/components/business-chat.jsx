'use client';

import * as React from "react";
import { Bot, Send } from 'lucide-react';
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Input } from "@/components/ui/input";
import ReactMarkdown from 'react-markdown'


export function BusinessChat() {
  const [messages, setMessages] = React.useState([]);
  const [businessType, setBusinessType] = React.useState("");
  const [inputValue, setInputValue] = React.useState("");
  const [isLoading, setIsLoading] = React.useState(false);

  const fetchAIResponse = async (query) => {
    setIsLoading(true);
    try {
      const response = await fetch('https://53a8-34-73-253-224.ngrok-free.app/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      });
      if (!response.ok) {
        const errorBody = await response.text();
        console.error('API Error:', response.status, errorBody);
        throw new Error(`API error ${response.status}: ${errorBody}`);
      }
      const data = await response.json();
      if (!data.response) {
        console.error('Invalid API response:', data);
        throw new Error('Invalid API response');
      }
      return data.response;
    } catch (error) {
      console.error('Error fetching AI response:', error);
      return `Sorry, I couldn't process your request at the moment. Error: ${error.message}`;
    } finally {
      setIsLoading(false);
    }
  };

  const handleSend = async () => {
    if (!inputValue.trim()) {
      alert("Please enter a message.");
      return;
    }
    if (!businessType) {
      alert("Please select a business type before sending a message.");
      return;
    }
    const newMessage = { text: `[${businessType}] ${inputValue}`, sender: 'user' };
    setMessages((prev) => [...prev, newMessage]);
    setInputValue("");
  };

  React.useEffect(() => {
    const getAIResponse = async () => {
      if (messages.length > 0 && messages[messages.length - 1].sender === 'user') {
        try {
          const aiResponse = await fetchAIResponse(messages[messages.length - 1].text);
          setMessages((prev) => [...prev, { text: aiResponse, sender: 'ai' }]);
        } catch (error) {
          console.error('Error in getAIResponse:', error);
          setMessages((prev) => [...prev, { text: 'Sorry, an error occurred while processing your request.', sender: 'ai' }]);
        }
      }
    };
    getAIResponse();
  }, [messages]);

  return (
    <div className="flex flex-col h-screen bg-slate-950 relative overflow-hidden">
      <div className="absolute top-0 left-0 w-96 h-96 bg-purple-500 rounded-full opacity-10 -translate-x-1/2 -translate-y-1/2" />
      <div className="absolute bottom-0 right-0 w-96 h-96 bg-purple-500 rounded-full opacity-10 translate-x-1/2 translate-y-1/2" />
      <div className="flex items-center justify-between p-4 border-b border-slate-800 relative z-10">
        <div className="flex items-center gap-2">
          <Bot className="w-5 h-5 text-slate-400" />
          <h1 className="text-lg font-semibold text-slate-200">Chat Assistant</h1>
        </div>
        <Select value={businessType} onValueChange={setBusinessType}>
          <SelectTrigger className="w-[200px] bg-slate-900 border-slate-800 text-slate-200">
            <SelectValue placeholder="Select nano business" />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value="food-truck">Food Truck</SelectItem>
            <SelectItem value="freelance-design">Freelance Design</SelectItem>
            <SelectItem value="home-bakery">Home Bakery</SelectItem>
            <SelectItem value="pet-sitting">Pet Sitting</SelectItem>
            <SelectItem value="handmade-crafts">Handmade Crafts</SelectItem>
          </SelectContent>
        </Select>
      </div>
      <div className="flex-1 overflow-auto p-4 space-y-4 relative z-10">
  {messages.map((message, index) => (
    <div
      key={index}
      className={`flex items-start gap-3 ${
        message.sender === 'user' ? 'justify-end' : 'justify-start'
      }`}
    >
      {message.sender === 'ai' && (
        <Avatar className="w-8 h-8">
          <AvatarImage src="/placeholder.svg" />
          <AvatarFallback>AI</AvatarFallback>
        </Avatar>
      )}
      <div
        className={`rounded-lg px-4 py-2 max-w-[80%] ${
          message.sender === 'user'
            ? 'bg-primary text-white'
            : 'bg-slate-800 text-slate-200'
        }`}
      >
        {message.sender === 'ai' ? (
          <ReactMarkdown>{message.text}</ReactMarkdown>
        ) : (
          message.text
        )}
      </div>
      {message.sender === 'user' && (
        <Avatar className="w-8 h-8">
          <AvatarImage src="/placeholder.svg" />
          <AvatarFallback>ME</AvatarFallback>
        </Avatar>
      )}
    </div>
  ))}
</div>
      <div className="p-4 border-t border-slate-800 relative z-10">
        <div className="flex gap-2 max-w-2xl mx-auto">
          <Input
            placeholder="Type your message..."
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                handleSend();
              }
            }}
            className="bg-slate-900 border-slate-800 text-slate-200"
          />
          <Button onClick={handleSend} className="shrink-0" disabled={isLoading}>
            <Send className="w-4 h-4" />
            <span className="sr-only">Send message</span>
          </Button>
        </div>
      </div>
    </div>
  );
}
