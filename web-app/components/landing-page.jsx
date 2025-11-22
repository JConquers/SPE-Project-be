'use client'

import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import Link from "next/link"
import { ArrowRight, BarChart3, Building2, Calendar, CheckCircle2, CreditCard, DollarSign, FileText, Users, Wallet } from 'lucide-react'
import { motion, useMotionValue, useSpring, useTransform } from "framer-motion"


export default function LandingPageComponent() {


  const [scrolled, setScrolled] = useState(false)
  const cardRef = useRef(null)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 50)
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll);
  }, [])

  const handleMouseMove = (e) => {
    if (cardRef.current) {
      const rect = cardRef.current.getBoundingClientRect()
      setMousePosition({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top,
      })
    }
  }

  const rotateX = useSpring(useTransform(useMotionValue(mousePosition.y), [0, 300], [10, -10]))
  const rotateY = useSpring(useTransform(useMotionValue(mousePosition.x), [0, 300], [-10, 10]))

  const features = [
    {
      icon: <CreditCard className="h-8 w-8" />,
      title: "Unique Nano-Enterprise ID",
      description: "Track and store data across systems with a unique identifier for each nano-enterprise"
    },
    {
      icon: <BarChart3 className="h-8 w-8" />,
      title: "Creditworthiness Score",
      description: "Predict timely repayment likelihood based on financial and socio-economic factors"
    },
    {
      icon: <CheckCircle2 className="h-8 w-8" />,
      title: "Aadhaar Verification",
      description: "Prevent fraud through India's Aadhaar system verification"
    },
    {
      icon: <Wallet className="h-8 w-8" />,
      title: "Loan Management",
      description: "Track current loan amounts and terms for better financial planning"
    },
    {
      icon: <DollarSign className="h-8 w-8" />,
      title: "Revenue Tracking",
      description: "Monitor annual revenue to assess loan repayment capability"
    },
    {
      icon: <Calendar className="h-8 w-8" />,
      title: "Payment Schedule",
      description: "Track monthly debt payments and manage repayment schedules"
    },
    {
      icon: <Building2 className="h-8 w-8" />,
      title: "Business Analytics",
      description: "Analyze years in business and stability indicators"
    },
    {
      icon: <Users className="h-8 w-8" />,
      title: "Workforce Insights",
      description: "Track employee count to assess income generation capacity"
    },
    {
      icon: <FileText className="h-8 w-8" />,
      title: "Business Classification",
      description: "Categorize enterprises by sector for specialized assessment"
    }
  ]

  return (
    (<div className="min-h-screen bg-[#0a0b1a] text-white">
      {/* Navigation */}
      <nav
        className={`fixed top-0 w-full z-50 transition-all duration-300 ${
          scrolled ? 'bg-[#0a0b1a]/80 backdrop-blur-md' : ''
        }`}>
        <div className="container mx-auto px-4 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <DollarSign className="h-8 w-8 text-purple-500" />
              <span className="text-xl font-bold">NanoCredit AI</span>
            </div>
            <div className="hidden md:flex items-center space-x-8">
              <Link
                href="#features"
                className="text-sm hover:text-purple-400 transition-colors">
                Features
              </Link>
              <Link href="#about" className="text-sm hover:text-purple-400 transition-colors">
                About
              </Link>
              <Link
                href="#contact"
                className="text-sm hover:text-purple-400 transition-colors">
                Contact
              </Link>
            </div>
            <div className="flex items-center space-x-4">
              <Link href="/login-page" variant="ghost" className="hidden md:inline-flex">
                Login
              </Link>
              <Button className="bg-purple-600 hover:bg-purple-700">
                Get Started <ArrowRight className="ml-2 h-4 w-4" />
              </Button>
            </div>
          </div>
        </div>
      </nav>
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 md:pt-48 md:pb-32">
        {/* Background Effects */}
        <div className="absolute inset-0 overflow-hidden">
          <div
            className="absolute w-[500px] h-[500px] -top-[250px] -left-[250px] bg-purple-500/10 rounded-full blur-3xl" />
          <div
            className="absolute w-[500px] h-[500px] -bottom-[250px] -right-[250px] bg-indigo-500/10 rounded-full blur-3xl" />
        </div>

        <div className="container mx-auto px-4 relative">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            className="text-center max-w-3xl mx-auto">
            <h1
              className="text-4xl md:text-6xl font-bold mb-6 bg-clip-text text-transparent bg-gradient-to-r from-purple-400 to-indigo-400">
              Smart Credit Solutions for Nano Enterprises
            </h1>
            <p className="text-xl md:text-2xl text-gray-400 mb-8">
              Empower your small business with AI-driven creditworthiness assessment and seamless loan management
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
              <Button size="lg" className="bg-purple-600 hover:bg-purple-700">
                Check Eligibility
              </Button>
              <Link href="/datapoints">
              <Button
                size="lg"
                variant="outline"
                className="border-purple-500 text-purple-400 hover:bg-purple-500/10">
                Learn More
              </Button>
              </Link>
            </div>
          </motion.div>
        </div>
      </section>
      {/* 3D Card Section */}
      <section className="py-20 relative">
        <div className="container mx-auto px-4">
          <div className="flex flex-col lg:flex-row items-center justify-between gap-12">
            <div className="lg:w-1/2">
              <h2 className="text-3xl md:text-4xl font-bold mb-6">Unique Nano-Enterprise ID</h2>
              <p className="text-xl text-gray-400 mb-8">
                Our innovative Unique ID system ensures secure and efficient tracking of your nano-enterprise across all our services.
              </p>
              <ul className="space-y-4">
                <li className="flex items-center">
                  <CheckCircle2 className="h-6 w-6 text-green-500 mr-2" />
                  <span>Secure and tamper-proof identification</span>
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-6 w-6 text-green-500 mr-2" />
                  <span>Seamless integration with all NanoCredit AI services</span>
                </li>
                <li className="flex items-center">
                  <CheckCircle2 className="h-6 w-6 text-green-500 mr-2" />
                  <span>Quick access to your business profile and credit history</span>
                </li>
              </ul>
            </div>
            <div className="lg:w-1/2">
              <motion.div
                ref={cardRef}
                className="w-[300px] h-[400px] bg-gradient-to-br from-purple-600 to-indigo-600 rounded-xl shadow-xl perspective-1000"
                style={{
                  rotateX: rotateX,
                  rotateY: rotateY,
                }}
                onMouseMove={handleMouseMove}
                onMouseLeave={() => {
                  rotateX.set(0)
                  rotateY.set(0)
                }}>
                <div className="w-full h-full flex flex-col justify-between p-6 text-white">
                  <div>
                    <h3 className="text-2xl font-bold mb-2">NanoCredit AI</h3>
                    <p className="text-sm opacity-75">Unique Enterprise ID</p>
                  </div>
                  <div>
                    <p className="text-3xl font-mono mb-2">NCAI-1234-5678-90AB</p>
                    <p className="text-sm opacity-75">Valid: 01/24 - 01/29</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold">John's Nano Enterprise</p>
                    <p className="text-xs opacity-75">Retail</p>
                  </div>
                </div>
              </motion.div>
            </div>
          </div>
        </div>
      </section>
      {/* Features Grid */}
      <section className="py-20 relative" id="features">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            whileInView={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5 }}
            viewport={{ once: true }}
            className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Comprehensive Credit Assessment</h2>
            <p className="text-gray-400 max-w-2xl mx-auto">
              Our AI-powered platform analyzes multiple factors to provide accurate creditworthiness predictions
            </p>
          </motion.div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20 }}
                whileInView={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5, delay: index * 0.1 }}
                viewport={{ once: true }}>
                <Card
                  className="bg-[#161832] border-neutral-800 hover:border-purple-500/50 transition-colors">
                  <CardContent className="p-6">
                    <div
                      className="w-12 h-12 rounded-lg bg-purple-500/10 flex items-center justify-center mb-4">
                      {feature.icon}
                    </div>
                    <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                    <p className="text-gray-200">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>
      {/* CTA Section */}
      <section className="py-20 relative">
        <div className="container mx-auto px-4">
          <Card className="bg-gradient-to-r from-purple-500/10 to-indigo-500/10 border-0">
            <CardContent className="p-8 md:p-12">
              <div className="max-w-2xl mx-auto text-center">
                <h2 className="text-3xl md:text-4xl font-bold mb-4">
                  Ready to Transform Your Business?
                </h2>
                <p className="text-gray-400 mb-8">
                  Join thousands of nano enterprises using our AI-powered credit assessment platform
                </p>
                <Button size="lg" className="bg-purple-600 hover:bg-purple-700">
                  Get Started Now <ArrowRight className="ml-2 h-4 w-4" />
                </Button>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>
    </div>)
  );
}