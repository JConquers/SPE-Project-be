'use client'

import { useState } from 'react'
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Badge } from "@/components/ui/badge"
import { Sparkles, Users, ShoppingBag, CreditCard, Home, Briefcase, Calendar, Brain, MapPin, UserCheck, Clock, GraduationCap, PiggyBank, Shield } from 'lucide-react'

export function CreditworthinessAssessmentComponent() {
  const [activeTab, setActiveTab] = useState("community")

  const datapoints = [
    {
      id: "community",
      title: "Community Reputation",
      icon: <Users className="h-6 w-6" />,
      dataType: "String/Rating",
      collection: "Informal interviews, AI sentiment analysis",
      description: "Assess the business's standing in the local community through interviews and social media analysis."
    },
    {
      id: "business",
      title: "Business Activity",
      icon: <ShoppingBag className="h-6 w-6" />,
      dataType: "Integer/Estimated Value",
      collection: "Observations, Computer vision",
      description: "Track customer footfall and sales activity through direct observation or AI-powered cameras."
    },
    {
      id: "credit",
      title: "Informal Credit",
      icon: <CreditCard className="h-6 w-6" />,
      dataType: "Boolean/Rating",
      collection: "Supplier interviews, NLP analysis",
      description: "Evaluate repayment habits through discussions with suppliers and analysis of informal contracts."
    },
    {
      id: "assets",
      title: "Asset Ownership",
      icon: <Home className="h-6 w-6" />,
      dataType: "List/String",
      collection: "On-site verification, Image recognition",
      description: "Catalog business assets through direct verification or AI-powered image analysis."
    },
    {
      id: "bills",
      title: "Bill Payment Patterns",
      icon: <CreditCard className="h-6 w-6" />,
      dataType: "Boolean/Date",
      collection: "Document verification, OCR analysis",
      description: "Analyze utility bill payment history using document scans and OCR technology."
    },
    {
      id: "market",
      title: "Local Market Standing",
      icon: <Briefcase className="h-6 w-6" />,
      dataType: "Rating/Integer",
      collection: "Competitor analysis, NLP of social media",
      description: "Assess market position through competitor analysis and social media sentiment."
    },
    {
      id: "welfare",
      title: "Welfare Program Participation",
      icon: <Shield className="h-6 w-6" />,
      dataType: "Boolean/List",
      collection: "Document verification, Government API integration",
      description: "Verify enrollment in support programs through documents and government databases."
    },
    {
      id: "family",
      title: "Family and Dependents",
      icon: <Users className="h-6 w-6" />,
      dataType: "Integer",
      collection: "Interviews, Demographic data analysis",
      description: "Understand household size and business involvement through interviews and local data."
    },
    {
      id: "seasonal",
      title: "Seasonal Impact",
      icon: <Calendar className="h-6 w-6" />,
      dataType: "String/Rating",
      collection: "Interviews, Historical data analysis",
      description: "Evaluate seasonal revenue changes using interviews and historical sales data."
    },
    {
      id: "behavioral",
      title: "Behavioral Indicators",
      icon: <Brain className="h-6 w-6" />,
      dataType: "Rating/Score",
      collection: "Psychometric tests, Chatbot assessments",
      description: "Assess risk-taking and problem-solving skills through tests and AI-powered chatbots."
    },
    {
      id: "location",
      title: "Business Location",
      icon: <MapPin className="h-6 w-6" />,
      dataType: "String/Rating",
      collection: "On-site visits, Satellite image analysis",
      description: "Evaluate business location quality through visits and AI analysis of geographical data."
    },
    {
      id: "relationships",
      title: "Customer Relationships",
      icon: <UserCheck className="h-6 w-6" />,
      dataType: "Rating/Count",
      collection: "Interviews, NLP of communication history",
      description: "Analyze customer and supplier relationships through interviews and communication history."
    },
    {
      id: "consistency",
      title: "Operational Consistency",
      icon: <Clock className="h-6 w-6" />,
      dataType: "Boolean",
      collection: "Regular visits, IoT sensor monitoring",
      description: "Monitor business operation consistency through visits or IoT sensors."
    },
    {
      id: "training",
      title: "Skill Development",
      icon: <GraduationCap className="h-6 w-6" />,
      dataType: "Boolean/List",
      collection: "Certificate verification, Database cross-referencing",
      description: "Verify training and certifications through documents and online databases."
    },
    {
      id: "debt",
      title: "Debt and Liabilities",
      icon: <CreditCard className="h-6 w-6" />,
      dataType: "Float/Integer",
      collection: "Interviews, NLP of financial records",
      description: "Assess outstanding debts through interviews and analysis of financial records."
    },
    {
      id: "savings",
      title: "Savings and Insurance",
      icon: <PiggyBank className="h-6 w-6" />,
      dataType: "Boolean/Float",
      collection: "Interviews, Bank API integration",
      description: "Evaluate savings and insurance through interviews and real-time financial data."
    },
  ]

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-b from-slate-950 to-indigo-950 justify-center items-center">
      <main className="flex-1 py-8 md:py-12 lg:py-16">
        <div className="container px-4 md:px-6">
          <div className="flex flex-col items-center justify-center space-y-6 text-center">
            <div className="space-y-4">
              <h1 className="text-3xl font-bold tracking-tighter text-white sm:text-4xl md:text-5xl lg:text-6xl/none">
                Creditworthiness Assessment
              </h1>
              <p className="mx-auto max-w-[700px] text-gray-400 md:text-xl">
                Explore the datapoints we use to assess creditworthiness for nano enterprises.
              </p>
            </div>
          </div>
  
          <Tabs defaultValue="community" className="mt-16" onValueChange={setActiveTab}>
            <TabsList className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-8 gap-4">
              {datapoints.map((datapoint) => (
                <TabsTrigger
                  key={datapoint.id}
                  value={datapoint.id}
                  className={`px-3 py-2 text-sm font-medium rounded transition-colors ${
                    activeTab === datapoint.id
                      ? 'bg-violet-600 text-white'
                      : 'bg-violet-600 text-white hover:bg-slate-700'
                  }`}
                >
                  {datapoint.title}
                </TabsTrigger>
              ))}
            </TabsList>

            {datapoints.map((datapoint) => (
              <TabsContent key={datapoint.id} value={datapoint.id} className="mt-28">
                <Card className="bg-slate-900 border-violet-400/20">
                  <CardHeader>
                    <div className="flex items-start space-x-4">
                      <div className="p-2 bg-violet-600 rounded-full flex-shrink-0">
                        {datapoint.icon}
                      </div>
                      <div>
                        <CardTitle className="text-2xl font-bold text-white">{datapoint.title}</CardTitle>
                      </div>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-4 pt-4">
                    <CardDescription className="text-gray-400 mb-6">
                      {datapoint.description}
                    </CardDescription>
                    <div className="flex flex-col space-y-4">
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary" className="bg-violet-600 text-white">
                          Data Type
                        </Badge>
                        <span className="text-gray-300">{datapoint.dataType}</span>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Badge variant="secondary" className="bg-indigo-600 text-white">
                          Collection Method
                        </Badge>
                        <span className="text-gray-300">{datapoint.collection}</span>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            ))}
          </Tabs>
        </div>
      </main>
      <footer className="w-full border-t border-gray-700 py-6 bg-slate-950">
        <div className="container flex flex-col items-center justify-between gap-6 md:h-24 md:flex-row md:py-0">
          <div className="flex items-center gap-4 px-8 md:px-0">
            <Sparkles className="h-6 w-6 text-violet-400" />
            <p className="text-center text-sm leading-loose text-gray-300 md:text-left">
              Built with ❤️ for Indian Nano Enterprises
            </p>
          </div>
          <div className="flex flex-col md:flex-row items-center gap-4 px-8 md:px-0">
            <Button variant="ghost" className="text-sm text-gray-300 hover:text-violet-400">
              Terms
            </Button>
            <Button variant="ghost" className="text-sm text-gray-300 hover:text-violet-400">
              Privacy
            </Button>
            <Button variant="ghost" className="text-sm text-gray-300 hover:text-violet-400">
              Contact
            </Button>
          </div>
        </div>
      </footer>
    </div>
  );
}  