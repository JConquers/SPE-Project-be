'use client';
import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Progress } from "@/components/ui/progress"
import { ResponsiveContainer, LineChart, Line, XAxis, YAxis, Tooltip } from 'recharts'
import Link from "next/link";
import { useRouter } from 'next/navigation'; 

import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { ScrollArea } from "@/components/ui/scroll-area"
import {
  Activity,
  BarChart3,
  Calendar,
  Clock,
  CreditCard,
  DollarSign,
  Download,
  HandCoins,
  Home,
  Settings,
  Share2,
  ShieldCheck,
  Sparkles,
  Star,
} from 'lucide-react';



// Placeholder chart component (replace with actual chart library component)
// const LineChart = ({ data, xAxis, yAxis }) => (
//   <div
//     className="w-full h-64 bg-gradient-to-r from-violet-500/20 to-indigo-500/20 rounded-lg flex items-center justify-center">
//     Line Chart Placeholder
//   </div>
// )



export function BorrowerDashboardComponent() {
  const [activeTab, setActiveTab] = useState("overview")
  const [scrolled, setScrolled] = useState(false)
  const [progress, setProgress] = useState(0)
  const [cardVisible, setCardVisible] = useState(false)
  const cardRef = useRef(null)
  const [mousePosition, setMousePosition] = useState({ x: 0, y: 0 })

  const [applicationData, setApplicationData] = useState({
    vendor_name: '',
    business_name: '',
    business_title: '',
    location: '',
    welfare_scheme: '',
    family_members: '',
    shop_location: '',
    government_ids: [{ type: '', number: '' }]
  })
  
  const [analysisResult, setAnalysisResult] = useState(null)

  const router = useRouter();

  useEffect(() => {
    const handleScroll = () => {
      setScrolled(window.scrollY > 50)
    }
    window.addEventListener('scroll', handleScroll)
    return () => window.removeEventListener('scroll', handleScroll);
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => setProgress(66), 500)
    return () => clearTimeout(timer);
  }, [])

  useEffect(() => {
    const timer = setTimeout(() => setCardVisible(true), 1000)
    return () => clearTimeout(timer);
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

  const handleMouseLeave = () => {
    setMousePosition({ x: 0, y: 0 })
  }

  const cardTransform = () => {
    if (!cardRef.current) return ''
    const { width, height } = cardRef.current.getBoundingClientRect()
    const { x, y } = mousePosition
    const tiltX = (y - height / 2) / 20
    const tiltY = (width / 2 - x) / 20
    return `perspective(1000px) rotateX(${tiltX}deg) rotateY(${tiltY}deg)`
  }

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch('http://127.0.0.1:5000/get-analysis')  // Replace with your actual API endpoint
        if (!response.ok) {
          throw new Error('Failed to fetch analysis data')
        }
        const data = await response.json()
  
        // Extract the fields from the response
        const { analysis_result, form_data } = data
        console.log('Analysis result:', analysis_result)
        console.log('Form data:', form_data)



        const parseAnalysisResult = (input) => {
          // Step 1: Remove the "Analysis result:" prefix
          const cleanedInput = input.replace("Analysis result:", "").trim();
      
          // Step 2: Split the input into lines
          const lines = cleanedInput.split("\n");
      
          // Step 3: Parse each line into key-value pairs
          const result = {};
          lines.forEach(line => {
              const [key, value] = line.split(/:(.+)/); // Split at the first colon
              if (key && value) {
                  result[key.trim()] = value.trim().replace(/^"|"$/g, ""); // Remove quotes if present
              }
          });
      
          return result;
      };


      const parsedResult = parseAnalysisResult(analysis_result);
      const parsedResultString = JSON.stringify(parsedResult);



        




        // console.log(parsedResult)
        // Set the state for form fields and analysis result
        console.log(parsedResult,"hello")
        console.log(parsedResultString)
        setAnalysisResult(parsedResultString)
        // console.log(analysisResult.customer_review,"helloi")
        setApplicationData({
          vendor_name: form_data.vendor_name  || '',
          business_name: form_data.business_name || '',
          business_title: form_data.business_title  || '',
          location: form_data.location,
          welfare_scheme: form_data.welfare_scheme,
          family_members: form_data.family_members,
          shop_location: form_data.shop_location,
          government_ids: form_data.government_ids,
        })
      } catch (error) {
        console.error('Error fetching data:', error)
      }
    }
  
    fetchData()
  }, [])  // Runs once on component mount

  return (
    (<div
      className="flex h-screen bg-gradient-to-b from-slate-950 to-indigo-950 text-white relative overflow-hidden">
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
        {/* Animated floating shapes */}
        <div className="absolute inset-0">
          <div
            className="absolute top-1/4 left-1/4 w-2 h-2 bg-violet-400/20 rounded-full animate-float1"></div>
          <div
            className="absolute top-3/4 left-1/2 w-3 h-3 bg-violet-400/30 rounded-full animate-float2"></div>
          <div
            className="absolute top-1/2 right-1/4 w-2 h-2 bg-indigo-400/20 rounded-full animate-float3"></div>
        </div>
      </div>
      {/* Sidebar */}
      <div className="w-64 bg-slate-900/50 backdrop-blur-sm p-4 z-10">
        <div className="flex items-center mb-8">
          <DollarSign className="mr-2 h-8 w-8 text-violet-400" />
          <h1 className="text-2xl font-bold">NanoCredit AI</h1>
        </div>
        <nav>
          <Button
            variant="ghost"
            className="w-full justify-start mb-2"
            onClick={() => setActiveTab("overview")}>
            <Home className="mr-2 h-4 w-4" /> Overview
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start mb-2"
            onClick={() => setActiveTab("loans")}>
            <HandCoins className="mr-2 h-4 w-4" /> My Loans
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start mb-2"
            onClick={() => setActiveTab("credit-score")}>
            <BarChart3 className="mr-2 h-4 w-4" /> Credit Score
          </Button>
          <Button
            variant="ghost"
            className="w-full justify-start mb-2"
            onClick={() => setActiveTab("data-card")}>
            <CreditCard className="mr-2 h-4 w-4" /> Credit Data Card
          </Button>
          <Button variant="ghost" className="w-full justify-start">
            <Settings className="mr-2 h-4 w-4" /> Settings
          </Button>
          {/* <Button variant="ghost" className="w-full justify-start">
            <Settings className="mr-2 h-4 w-4" /> Survey
            
          </Button> */}
          <Button
            variant="ghost"
            className="w-full justify-start mb-2"
            // onClick={() => router.push('/repayment_chat')}>
            onClick={() => router.push('/chat-dashboard')}>

            <CreditCard className="mr-2 h-4 w-4" /> Survey
          </Button>


          {/* <Button variant="ghost" className="w-full justify-start">
            <Settings className="mr-2 h-4 w-4" /> Ask Ai
          </Button> */}

<Button
            variant="ghost"
            className="w-full justify-start mb-2"
            // onClick={() => setActiveTab("data-card")}>
            onClick={() => router.push('/repayment_chat')}>

            <CreditCard className="mr-2 h-4 w-4" /> Ask Ai
          </Button>

          {/* <Button variant="ghost" className="w-full justify-start">
            <HandCoins className="mr-2 h-4 w-4" /> Personalized Loan
          </Button> */}

      <Button
            variant="ghost"
            className="w-full justify-start mb-2"
            // onClick={() => setActiveTab("chat-card")}>
            onClick={() => router.push('/custom')}>

            <CreditCard className="mr-2 h-4 w-4" /> Personalized Loan
          </Button>
          
        </nav>
      </div>
      {/* Main Content */}
      <div className="flex-1 p-8 overflow-auto">
        <div className="flex justify-between items-center mb-8">
          <h2 className="text-3xl font-bold">Borrower Dashboard</h2>
          <div className="flex items-center">
            <Input
              type="search"
              placeholder="Search..."
              className="mr-4 bg-slate-800/50 border-slate-700 text-white placeholder:text-slate-400" />
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="outline" className="border-slate-700 bg-slate-800/50 text-white">{applicationData.business_name}</Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent className="bg-slate-800 border-slate-700">
                <DropdownMenuLabel>My Account</DropdownMenuLabel>
                <DropdownMenuSeparator />
                <DropdownMenuItem>Profile</DropdownMenuItem>
                <DropdownMenuItem>Billing</DropdownMenuItem>
                <DropdownMenuItem>Team</DropdownMenuItem>
                <DropdownMenuItem>Subscription</DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </div>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-4">
          <TabsList className="bg-slate-800/50 border border-slate-700">
            <TabsTrigger value="overview" className="data-[state=active]:bg-violet-600">Overview</TabsTrigger>
            <TabsTrigger value="loans" className="data-[state=active]:bg-violet-600">My Loans</TabsTrigger>
            <TabsTrigger value="credit-score" className="data-[state=active]:bg-violet-600">Credit Score</TabsTrigger>
            <TabsTrigger value="data-card" className="data-[state=active]:bg-violet-600">Credit Data Card</TabsTrigger>
          </TabsList>
          <TabsContent value="overview" className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Active Loans
                  </CardTitle>
                  <CreditCard className="h-4 w-4 text-violet-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">2</div>
                  <p className="text-xs text-slate-400">
                    Total amount: ₹45,000
                  </p>
                </CardContent>
              </Card>
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Credit Score
                  </CardTitle>
                  <BarChart3 className="h-4 w-4 text-violet-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">621</div>
                  <p className="text-xs text-slate-400">
                    +15 from last month
                  </p>
                </CardContent>
              </Card>
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Repayment Progress
                  </CardTitle>
                  <Activity className="h-4 w-4 text-violet-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">75%</div>
                  <Progress value={75} className="mt-2" />
                </CardContent>
              </Card>
              <Card className="bg-slate-800/50 border-slate-700">
                <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                  <CardTitle className="text-sm font-medium">
                    Next Due Date
                  </CardTitle>
                  <Calendar className="h-4 w-4 text-violet-400" />
                </CardHeader>
                <CardContent>
                  <div className="text-2xl font-bold">15 Jun 2024</div>
                  <p className="text-xs text-slate-400">
                    Amount due: ₹5,000
                  </p>
                </CardContent>
              </Card>
            </div>
            <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-7">
              <Card className="col-span-4 bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle>Loan Repayment History</CardTitle>
                </CardHeader>
                <CardContent className="pl-2">
                <div className="h-[500px]">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart
                        data={[
                          { month: 'Jan', amount: 5000 },
                          { month: 'Feb', amount: 10000 },
                          { month: 'Mar', amount: 25000 },
                          { month: 'Apr', amount: 20000 },
                          { month: 'May', amount: 45000 },
                          { month: 'Jun', amount: 30000 },
                        ]}
                      >
                        <XAxis
                          dataKey="month"
                          stroke="#888888"
                          fontSize={12}
                          tickLine={false}
                          axisLine={false}
                        />
                        <YAxis
                          stroke="#888888"
                          fontSize={12}
                          tickLine={false}
                          axisLine={false}
                          tickFormatter={(value) => `₹${value}`}
                        />
                        <Tooltip />
                        <Line
                          type="monotone"
                          dataKey="amount"
                          stroke="#8884d8"
                          strokeWidth={2}
                          dot={{ fill: "#8884d8", strokeWidth: 2 }}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                </CardContent>
              </Card>
              <Card className="col-span-3 bg-slate-800/50 border-slate-700">
                <CardHeader>
                  <CardTitle>Important Business Factors</CardTitle>
                  <CardDescription className="text-slate-400">
                    Factors influencing your credit score
                  </CardDescription>
                </CardHeader>
                <CardContent>
    <div className="space-y-4">
      {/* Check if analysisResult is valid */}
      {analysisResult ? (
        Object.entries(JSON.parse(analysisResult)).map(([key, value]) => (
          <div key={key} className="flex items-center">
            <div className="mr-2 text-violet-400">
              {/* Placeholder for an icon */}
              <ShieldCheck className="w-4 h-4" />
            </div>
            <div className="flex-grow">
              <div className="flex justify-between mb-1">
                <span className="text-sm font-medium capitalize">
                  {key.replace(/_/g, ' ')}
                </span>
                <span className="text-sm font-medium text-gray-300">{value}</span>
              </div>
              {/* Optionally include a progress bar */}
              <Progress value={70} className="h-2" />
            </div>
          </div>
        ))
      ) : (
        <div className="text-gray-400">No data available</div>
      )}
    </div>
  </CardContent>
              </Card>
            </div>
          </TabsContent>
          <TabsContent value="loans" className="space-y-4">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle>My Active Loans</CardTitle>
                <CardDescription className="text-slate-400">
                  Details of your current loans
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[400px]">
                  {[
                    { id: 1, amount: 25000, purpose: "Business Expansion", dueDate: "2024-07-15", progress: 60 },
                    { id: 2, amount: 20000, purpose: "Equipment Purchase", dueDate: "2024-08-30", progress: 40 },
                  ].map((loan) => (
                    <div
                      key={loan.id}
                      className="mb-4 p-4 border rounded-lg border-slate-700 bg-slate-800/30">
                      <div className="flex justify-between items-center mb-2">
                        <h3 className="font-semibold">Loan #{loan.id}</h3>
                        <span className="text-sm text-slate-400">Due: {loan.dueDate}</span>
                      </div>
                      <p className="mb-2">Amount: ₹{loan.amount}</p>
                      <p className="mb-2">Purpose: {loan.purpose}</p>
                      <div className="flex items-center">
                        <Progress value={loan.progress} className="flex-grow mr-2" />
                        <span className="text-sm">{loan.progress}%</span>
                      </div>
                    </div>
                  ))}
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="credit-score" className="space-y-4">
            <Card className="bg-slate-800/50 border-slate-700">
              <CardHeader>
                <CardTitle>Your Credit Score</CardTitle>
                <CardDescription className="text-slate-400">
                  Understanding your credit worthiness
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="flex justify-center items-center mb-8">
                  <div className="relative w-48 h-48">
                    <svg className="w-full h-full" viewBox="0 0 100 100">
                      <circle
                        className="text-slate-700 stroke-current"
                        strokeWidth="10"
                        cx="50"
                        cy="50"
                        r="40"
                        fill="transparent"></circle>
                      <circle
                        className="text-violet-500 stroke-current"
                        strokeWidth="10"
                        strokeLinecap="round"
                        cx="50"
                        cy="50"
                        r="40"
                        fill="transparent"
                        strokeDasharray="251.2"
                        strokeDashoffset="62.8"></circle>
                    </svg>
                    <div className="absolute inset-0 flex items-center justify-center">
                      <span className="text-4xl font-bold">621</span>
                    </div>
                  </div>
                </div>
                <div className="space-y-4">
                  {[
                     { name: "Community Reputation", score: 95, icon: <ShieldCheck className="w-4 h-4" /> },
                     { name: "Business Activity Observations", score: 85, icon: <BarChart3 className="w-4 h-4" /> },
                     { name: "Informal Credit Relationships", score: 70, icon: <Clock className="w-4 h-4" /> },
                     { name: "Asset Ownership", score: 80, icon: <Star className="w-4 h-4" /> },
                     { name: "Utility and Bill Payment Patterns", score: 90, icon: <Sparkles className="w-4 h-4" /> },
                     { name: "Local Market Standing", score: 88, icon: <ShieldCheck className="w-4 h-4" /> },
                     { name: "Participation in Welfare or Support Programs", score: 78, icon: <BarChart3 className="w-4 h-4" /> },
                     { name: "Family and Dependents", score: 82, icon: <Clock className="w-4 h-4" /> },
                     { name: "Seasonal and Environmental Impact", score: 65, icon: <Star className="w-4 h-4" /> },
                     { name: "Behavioral Indicators", score: 90, icon: <Sparkles className="w-4 h-4" /> },
                     { name: "Business Location and Infrastructure", score: 75, icon: <ShieldCheck className="w-4 h-4" /> },
                     { name: "Customer and Supplier Relationships", score: 80, icon: <BarChart3 className="w-4 h-4" /> },
                     { name: "Observed Consistency", score: 85, icon: <Clock className="w-4 h-4" /> },
                     { name: "Training and Skill Development", score: 88, icon: <Star className="w-4 h-4" /> },
                     { name: "Debt and Liabilities", score: 70, icon: <Sparkles className="w-4 h-4" /> },
                     { name: "Savings and Insurance", score: 90, icon: <ShieldCheck className="w-4 h-4" /> },
                     { name: "Aadhaar Card", score: 95, icon: <BarChart3 className="w-4 h-4" /> },
                     { name: "Ration Card", score: 85, icon: <Clock className="w-4 h-4" /> },
                     { name: "Voter ID Card", score: 80, icon: <Star className="w-4 h-4" /> },
                  ].map((factor) => (
                    <div key={factor.name} className="flex items-center">
                      <div className="mr-2 text-violet-400">{factor.icon}</div>
                      <div className="flex-grow">
                        <div className="flex justify-between mb-1">
                          <span className="text-sm font-medium">{factor.name}</span>
                          <span className="text-sm font-medium">{factor.score}%</span>
                        </div>
                        <Progress value={factor.score} className="h-2" />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </TabsContent>
          <TabsContent value="data-card" className="space-y-4">
            <div className="flex flex-col min-h-[calc(100vh-8rem)]">
              <section className="w-full py-12 md:py-24 lg:py-32 relative overflow-hidden">
                <div className="container px-4 md:px-6 relative">
                  <div className="grid gap-6 lg:grid-cols-2 items-center">
                    <div className="space-y-4">
                      <h2
                        className="text-3xl font-bold tracking-tighter text-white md:text-4xl/tight">
                        Your Credit Data Card
                      </h2>
                      <p className="text-slate-400 md:text-xl">
                        Your key to secure, seamless access to financial opportunities. Share it responsibly with authorized bankers or institutions.
                      </p>
                      <div className="space-y-2">
                        <p className="text-lg font-semibold text-violet-400">Your Unique ID</p>
                        <p className="text-2xl font-bold text-white">NCAI-1234-5678-90AB</p>
                      </div>
                      <div className="space-y-2">
                        <p className="text-sm text-slate-400">
                          • Ensures data privacy and secure sharing
                        </p>
                        <p className="text-sm text-slate-400">
                          • Allows controlled access to your financial information
                        </p>
                        <p className="text-sm text-slate-400">
                          • Simplifies the loan application process
                        </p>
                      </div>
                      <div className="flex flex-col sm:flex-row gap-2">
                        <Button className="bg-violet-600 text-white hover:bg-violet-700">
                          <Download className="mr-2 h-4 w-4" /> Download Card
                        </Button>
                        <Button
                          variant="outline"
                          className="text-violet-400 border-violet-400 hover:bg-violet-400/10">
                          <Share2 className="mr-2 h-4 w-4" /> Share Card
                        </Button>
                      </div>
                    </div>
                    <div
                      className="relative h-[400px] w-full max-w-md mx-auto"
                      onMouseMove={handleMouseMove}
                      onMouseLeave={handleMouseLeave}>
                      <div
                        ref={cardRef}
                        className={`absolute inset-0 transition-all duration-300 ease-out ${cardVisible ? 'opacity-100 translate-x-0' : 'opacity-0 translate-x-full'}`}
                        style={{ transform: cardTransform() }}>
                        <Card
                          className="w-full h-full overflow-hidden shadow-xl relative bg-gradient-to-br from-violet-600 to-indigo-600">
                          {/* Decorative card background */}
                          <div className="absolute inset-0 overflow-hidden">
                            <svg
                              className="absolute w-[600px] h-[600px] -top-[200px] -right-[200px] text-white/5"
                              viewBox="0 0 600 600">
                              <path
                                d="M300,300 Q450,150 600,300 T900,300"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2" />
                              <path
                                d="M300,350 Q450,200 600,350 T900,350"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2" />
                              <path
                                d="M300,400 Q450,250 600,400 T900,400"
                                fill="none"
                                stroke="currentColor"
                                strokeWidth="2" />
                            </svg>
                            <div
                              className="absolute top-0 left-0 w-full h-full bg-gradient-to-br from-white/10 via-transparent to-transparent"></div>
                          </div>
                          <CardContent className="p-6 flex flex-col justify-between h-full relative">
                            <div className="space-y-2">
                              <div className="flex items-center space-x-2">
                                <CreditCard className="h-12 w-12 text-white" />
                                <div className="w-8 h-8 rounded-full bg-white/20 animate-pulse"></div>
                              </div>
                              <h3 className="text-2xl font-bold text-white mt-4">MyNaNoCred Vault</h3>
                              <p className="text-lg text-white/80">Credit Data Card</p>
                            </div>
                            <div className="space-y-4">
                              <div className="space-y-1">
                                <p className="text-sm text-white/60">Card Holder</p>
                                <p className="text-lg font-semibold text-white">John Doe</p>
                              </div>
                              <div className="space-y-1">
                                <p className="text-sm text-white/60">Unique ID</p>
                                <p className="text-lg font-semibold text-white font-mono">NCAI-1234-5678-90AB</p>
                              </div>
                            </div>
                            {/* Decorative card elements */}
                            <div className="absolute bottom-0 right-0 w-32 h-32">
                              <svg viewBox="0 0 100 100" className="w-full h-full text-white/5">
                                <circle cx="50" cy="50" r="40" fill="none" stroke="currentColor" strokeWidth="1" />
                                <circle cx="50" cy="50" r="35" fill="none" stroke="currentColor" strokeWidth="1" />
                                <circle cx="50" cy="50" r="30" fill="none" stroke="currentColor" strokeWidth="1" />
                              </svg>
                            </div>
                          </CardContent>
                        </Card>
                      </div>
                    </div>
                  </div>
                </div>
              </section>
            </div>
          </TabsContent>
        </Tabs>
      </div>
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