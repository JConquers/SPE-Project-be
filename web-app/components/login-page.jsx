'use client'

import { useState, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Loader2, ShieldCheck, Coins, Link } from 'lucide-react'
import { useRouter } from 'next/navigation'; 

export  function LoginPageComponent () {
  const [step, setStep] = useState('aadhaar')
  const [isLoading, setIsLoading] = useState(false)
  const otpInputs = useRef([])
  const router = useRouter()

  const handleAadhaarSubmit = (e) => {
    e.preventDefault()
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
      setStep('otp')
    }, 1500)
  }

  const handleOtpSubmit = (e) => {
    e.preventDefault()
    setIsLoading(true)
    setTimeout(() => {
      setIsLoading(false)
      // Handle successful login here
         router.push('/application-form');
    }, 1500)
  }

  const handleOtpChange = (index, value) => {
    if (value.length === 1 && index < 5) {
      otpInputs.current[index + 1]?.focus()
    }
  }

  return (
    (<div
      className="min-h-screen flex items-center justify-center bg-gradient-to-b from-slate-950 to-indigo-950 p-4 relative overflow-hidden">
      {/* Decorative background elements */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <svg
          className="absolute w-[800px] h-[800px] -top-[400px] -left-[400px] text-violet-500/10"
          viewBox="0 0 800 800">
          <circle cx="400" cy="400" r="300" fill="currentColor" />
        </svg>
        <svg
          className="absolute w-[800px] h-[800px] -bottom-[400px] -right-[400px] text-indigo-500/10"
          viewBox="0 0 800 800">
          <circle cx="400" cy="400" r="300" fill="currentColor" />
        </svg>
      </div>
      {/* Animated floating shapes */}
      <div className="absolute inset-0 pointer-events-none">
        <div
          className="absolute top-1/4 left-1/4 w-2 h-2 bg-violet-400/20 rounded-full animate-float1"></div>
        <div
          className="absolute top-3/4 left-1/2 w-3 h-3 bg-violet-400/30 rounded-full animate-float2"></div>
        <div
          className="absolute top-1/2 right-1/4 w-2 h-2 bg-indigo-400/20 rounded-full animate-float3"></div>
      </div>
      {/* Background coins */}
      <div className="absolute inset-0 z-0 opacity-5">
        {[...Array(20)].map((_, i) => (
          <Coins
            key={i}
            className="absolute text-white"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              fontSize: `${Math.random() * 40 + 20}px`,
              transform: `rotate(${Math.random() * 360}deg)`,
            }} />
        ))}
      </div>
      <Card
        className="w-full max-w-md border-slate-800 bg-slate-900/50 backdrop-blur-sm text-white relative z-10">
        <CardHeader className="space-y-1 text-center">
          <CardTitle className="text-3xl font-bold tracking-tight">NanoCredit AI</CardTitle>
          <CardDescription className="text-slate-400">
            {step === 'aadhaar' ? 'Enter your Aadhaar number to continue' : 'Enter the OTP sent to your registered mobile'}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {step === 'aadhaar' ? (
            <form onSubmit={handleAadhaarSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="aadhaar" className="text-slate-200">
                  Aadhaar Number
                </Label>
                <Input
                  id="aadhaar"
                  placeholder="XXXX XXXX XXXX"
                  required
                  maxLength={14}
                  className="bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-500" />
              </div>
              <Button
                type="submit"
                className="w-full bg-violet-600 hover:bg-violet-700 text-white"
                disabled={isLoading}>
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Verifying
                  </>
                ) : (
                  'Continue'
                )}
              </Button>
            </form>
          ) : (
            <form onSubmit={handleOtpSubmit} className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="otp" className="text-slate-200">
                  OTP
                </Label>
                <div className="flex justify-between">
                  {[...Array(6)].map((_, index) => (
                    <Input
                      key={index}
                      type="text"
                      maxLength={1}
                      className="w-12 h-12 text-center bg-slate-800/50 border-slate-700 text-white"
                      ref={el => otpInputs.current[index] = el}
                      onChange={e => handleOtpChange(index, e.target.value)} />
                  ))}
                </div>
              </div>
          
            <Button
              type="submit"
              className="w-full bg-violet-600 hover:bg-violet-700 text-white"
              disabled={isLoading}
            >
              {isLoading ? (
                <>
                  <Loader2 className="mr-2 h-4 w-4 animate-spin" /> Verifying
                </>
              ) : (
                  'Login'
                
              )}
            </Button>
          
            </form>
          )}
        </CardContent>
        <CardFooter className="flex justify-center">
          <div className="flex items-center text-sm text-slate-400">
            <ShieldCheck className="mr-2 h-4 w-4" /> Secure verification powered by UIDAI
          </div>
        </CardFooter>
      </Card>
      <style jsx>{`
        @keyframes float1 {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-10px); }
        }
        @keyframes float2 {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-15px); }
        }
        @keyframes float3 {
          0%, 100% { transform: translateY(0px); }
          50% { transform: translateY(-20px); }
        }
        .animate-float1 {
          animation: float1 3s ease-in-out infinite;
        }
        .animate-float2 {
          animation: float2 4s ease-in-out infinite;
        }
        .animate-float3 {
          animation: float3 5s ease-in-out infinite;
        }
      `}</style>
    </div>)
  );
}