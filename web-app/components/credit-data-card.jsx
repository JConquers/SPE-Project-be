'use client'

import { useState, useRef, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Download, Share2, CreditCard } from 'lucide-react'

export function CreditDataCardComponent() {
  const [cardData, setCardData] = useState({
    name: '',
    loanType: '',
    amount: '',
    id: ''
  })
  const cardRef = useRef(null)

  const generateUniqueId = () => {
    const timestamp = Date.now().toString(36)
    const random = Math.random().toString(36).substring(2, 6)
    return `NCAI-${timestamp.slice(-4)}-${random}-${Math.random().toString(36).slice(-4)}`.toUpperCase();
  }

  const handleSubmit = () => {
    setCardData(prev => ({
      ...prev,
      id: generateUniqueId()
    }))
  }

  const handleShare = async () => {
    try {
      await navigator.share({
        title: 'Nano Loan Data Card',
        text: `My Credit Data Card ID: ${cardData.id}`,
        url: window.location.href
      })
    } catch (err) {
      console.log('Sharing failed', err)
    }
  }

  useEffect(() => {
    const card = cardRef.current
    if (!card) return

    const handleMouseMove = (e) => {
      const rect = card.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top

      const centerX = rect.width / 2
      const centerY = rect.height / 2

      const rotateX = (y - centerY) / 10
      const rotateY = (centerX - x) / 10

      card.style.transform = `perspective(1000px) rotateX(${rotateX}deg) rotateY(${rotateY}deg) scale3d(1.05, 1.05, 1.05)`
    }

    const handleMouseLeave = () => {
      card.style.transform = 'perspective(1000px) rotateX(0deg) rotateY(0deg) scale3d(1, 1, 1)'
    }

    card.addEventListener('mousemove', handleMouseMove)
    card.addEventListener('mouseleave', handleMouseLeave)

    return () => {
      card.removeEventListener('mousemove', handleMouseMove)
      card.removeEventListener('mouseleave', handleMouseLeave)
    };
  }, [])

  return (
    (<div className="min-h-screen bg-[#0a0627] text-white p-6">
      <div
        className="max-w-6xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
        <div className="space-y-6">
          <div>
            <h1 className="text-4xl font-bold mb-4">Your Loan Data Card</h1>
            <p className="text-gray-400">
              Your key to secure, seamless access to financial opportunities. Share it responsibly with authorized bankers or institutions.
            </p>
          </div>

          <Card className="bg-[#1a1040]/50 border-purple-500/20">
            <CardHeader>
              <CardTitle>Generate Your Card</CardTitle>
              <CardDescription className="text-gray-400">Fill in your details to create your credit data card</CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <Input
                  id="name"
                  value={cardData.name}
                  onChange={(e) => setCardData(prev => ({ ...prev, name: e.target.value }))}
                  className="bg-[#2a1f4d] border-purple-500/20"
                  placeholder="Enter your full name" />
              </div>
              <div className="space-y-2">
                <Label htmlFor="loanType">Loan Type</Label>
                <Select
                  onValueChange={(value) => setCardData(prev => ({ ...prev, loanType: value }))}>
                  <SelectTrigger className="bg-[#2a1f4d] border-purple-500/20">
                    <SelectValue placeholder="Select loan type" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="personal">Personal Loan</SelectItem>
                    <SelectItem value="business">Business Loan</SelectItem>
                    <SelectItem value="education">Education Loan</SelectItem>
                    <SelectItem value="home">Home Loan</SelectItem>
                  </SelectContent>
                </Select>
              </div>
              <div className="space-y-2">
                <Label htmlFor="amount">Loan Amount</Label>
                <Input
                  id="amount"
                  type="number"
                  value={cardData.amount}
                  onChange={(e) => setCardData(prev => ({ ...prev, amount: e.target.value }))}
                  className="bg-[#2a1f4d] border-purple-500/20"
                  placeholder="Enter loan amount" />
              </div>
            </CardContent>
            <CardFooter>
              <Button
                onClick={handleSubmit}
                className="w-full bg-purple-600 hover:bg-purple-700">
                Generate Card
              </Button>
            </CardFooter>
          </Card>

          {cardData.id && (
            <div className="space-y-4">
              <div>
                <h2 className="text-xl font-semibold mb-2">Your Unique ID</h2>
                <p className="text-purple-400 font-mono text-lg">{cardData.id}</p>
              </div>
              <div className="flex gap-4">
                <Button
                  variant="outline"
                  className="bg-purple-600/10 border-purple-500/30 hover:bg-purple-600/20"
                  onClick={() => {
                    // In a real app, implement card download functionality
                    console.log('Downloading card...')
                  }}>
                  <Download className="w-4 h-4 mr-2" />
                  Download Card
                </Button>
                <Button
                  variant="outline"
                  className="bg-purple-600/10 border-purple-500/30 hover:bg-purple-600/20"
                  onClick={handleShare}>
                  <Share2 className="w-4 h-4 mr-2" />
                  Share Card
                </Button>
              </div>
              <ul className="space-y-2 text-sm text-gray-400">
                <li>• Ensures data privacy and secure sharing</li>
                <li>• Allows controlled access to your financial information</li>
                <li>• Simplifies the loan application process</li>
              </ul>
            </div>
          )}
        </div>

        <div
          className="h-[400px] lg:h-[600px] w-full flex items-center justify-center">
          <div
            ref={cardRef}
            className="w-full max-w-md aspect-[1.6/1] bg-gradient-to-br from-purple-600 to-purple-800 text-white p-6 rounded-xl shadow-xl transition-transform duration-200 ease-out"
            style={{ transformStyle: 'preserve-3d' }}>
            <div className="flex flex-col h-full justify-between">
              <div className="flex justify-between items-start">
                <div className="space-y-1">
                  <p className="text-sm font-semibold">MyNaNoCred Vault</p>
                  <p className="text-xs">Credit Data Card</p>
                </div>
                <CreditCard className="w-10 h-10" />
              </div>
              <div className="space-y-4">
                <div>
                  <p className="text-xs">Card Holder</p>
                  <p className="font-semibold">{cardData.name || 'John Doe'}</p>
                </div>
                <div>
                  <p className="text-xs">Unique ID</p>
                  <p className="font-mono text-sm">{cardData.id || 'NCAI-1234-5678-90AB'}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>)
  );
}