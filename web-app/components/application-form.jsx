'use client';

import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Textarea } from '@/components/ui/textarea'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select'
import { ArrowLeft, ArrowRight, Share2 } from 'lucide-react'
import { useRouter } from 'next/navigation'

export function ApplicationFormComponent() {
  const [currentPage, setCurrentPage] = useState(0)
  const [loading, setLoading] = useState(false)
  const router = useRouter()
  const [selectedImage, setSelectedImage] = useState(null)
  const [formData, setFormData] = useState({
    vendorName: '',
    businessName: '',
    businessTitle: '',
    location: '',
    welfareScheme: '',
    familyMembers: '',
    shopLocation: '',
    governmentIds: [{ type: '', number: '' }]
  })

  const formPages = [
    'Personal Information',
    'Business Details',
    'Family Information',
    'Government IDs',
    'Additional Information'
  ]

  // --- FIX: Added '?? \'\'' to prevent undefined from any Input/Textarea ---
  const handleInputChange = (e) => {
    const { name, value } = e.target
    setFormData(prevData => ({
      ...prevData,
      [name]: value ?? '' // Guarantees the value is never undefined
    }))
  }

  // --- FIX: Added '?? \'\'' to prevent undefined from Welfare Scheme Select ---
  const handleSelectChange = (name, value) => {
    setFormData(prevData => ({
      ...prevData,
      [name]: value ?? '' // Guarantees the value is never undefined
    }));
  }

  // --- FIX: Added '?? \'\'' to prevent undefined from Government ID Select ---
  const handleGovIdChange = (field, value) => {
    setFormData(prevData => ({
      ...prevData,
      governmentIds: [{ ...prevData.governmentIds[0], [field]: value ?? '' }] // Guarantees the value is never undefined
    }))
  }

  const handleImageUpload = (event) => {
    const file = event.target.files[0]
    if (file) {
      setSelectedImage(file)
    }
  }

  const handleNext = () => {
    if (currentPage < formPages.length - 1) {
      setCurrentPage(currentPage + 1)
    }
  }

  const handleBack = () => {
    if (currentPage > 0) {
      setCurrentPage(currentPage - 1)
    }
  }

  const handleSubmit = async () => {
    setLoading(true)

    try {
      const formDataToSend = new FormData()

      Object.keys(formData).forEach(key => {
        if (key === 'governmentIds') {
          formDataToSend.append(key, JSON.stringify(formData[key]))
        } else {
          formDataToSend.append(key, formData[key])
        }
      })

      if (selectedImage) {
        formDataToSend.append('image', selectedImage, selectedImage.name)
      } else {
        throw new Error('No image selected')
      }

      const response = await fetch('http://127.0.0.1:5000/analyze', {
        method: 'POST',
        body: formDataToSend,
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const result = await response.json()
      console.log('Form submitted successfully:', result)
      alert('Form and image submitted successfully!')
      router.push('/dashboard')
    } catch (error) {
      console.error('Error submitting form:', error)
      alert(`Failed to submit form and/or image. Error: ${error.message}`)
    } finally {
      setLoading(false)
    }
  }

  const renderFormPage = () => {
    switch (currentPage) {
      case 0:
        return (<>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="vendorName" className="text-white">Vendor Name</Label>
              <Input
                id="vendorName"
                name="vendorName"
                value={formData.vendorName}
                onChange={handleInputChange}
                placeholder="Enter your full name"
                className="bg-white text-black placeholder-gray-500" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="businessName" className="text-white">Business Name</Label>
              <Input
                id="businessName"
                name="businessName"
                value={formData.businessName}
                onChange={handleInputChange}
                placeholder="Enter your business name"
                className="bg-white text-black placeholder-gray-500" />
            </div>
          </div>
        </>);
      case 1:
        return (<>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="businessTitle" className="text-white">Business Title</Label>
              <Input
                id="businessTitle"
                name="businessTitle"
                value={formData.businessTitle}
                onChange={handleInputChange}
                placeholder="Enter your business title"
                className="bg-white text-black placeholder-gray-500" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="location" className="text-white">Location</Label>
              <Input
                id="location"
                name="location"
                value={formData.location}
                onChange={handleInputChange}
                placeholder="Enter your business location"
                className="bg-white text-black placeholder-gray-500" />
            </div>
          </div>
        </>);
      case 2:
        return (<>
          <div className="space-y-4">
            <div className="space-y-2">
              <label htmlFor="welfare-scheme">Select Welfare Scheme:</label>
              <Select value={formData.welfareScheme} onValueChange={(value) => handleSelectChange('welfareScheme', value)}>
                <SelectTrigger id="welfare-scheme">
                  <SelectValue placeholder="Select a Scheme" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="pmmy">Pradhan Mantri Mudra Yojana (PMMY)</SelectItem>
                  <SelectItem value="kcc">Kisan Credit Card (KCC)</SelectItem>
                  <SelectItem value="psbloans">PSB Loans in 59 Minutes</SelectItem>
                  <SelectItem value="income-assessment">Income Assessment</SelectItem>
                  <SelectItem value="employment-verification">Employment Verification</SelectItem>
                  <SelectItem value="self-help-group">Self Help Group (SHG) Membership</SelectItem>
                  <SelectItem value="standup-india">Stand-Up India Scheme</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="familyMembers" className="text-white">Number of Family Members</Label>
              <Input
                id="familyMembers"
                name="familyMembers"
                type="number"
                value={formData.familyMembers}
                onChange={handleInputChange}
                placeholder="Enter number of family members"
                className="bg-white text-black placeholder-gray-500" />
            </div>
          </div>
        </>);
      case 3:
        return (<>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="governmentIdType" className="text-white">Government ID Type</Label>
              <Select
                onValueChange={(value) => handleGovIdChange('type', value)}
                value={formData.governmentIds[0].type}>
                <SelectTrigger className="bg-white text-black">
                  <SelectValue placeholder="Select ID type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="aadhar">Aadhar Card</SelectItem>
                  <SelectItem value="pan">PAN Card</SelectItem>
                  <SelectItem value="voter">Voter ID</SelectItem>
                  <SelectItem value="driving">Driving License</SelectItem>
                </SelectContent>
              </Select>
            </div>
            <div className="space-y-2">
              <Label htmlFor="governmentIdNumber" className="text-white">Government ID Number</Label>
              <Input
                id="governmentIdNumber"
                name="governmentIds[0].number"
                value={formData.governmentIds[0].number}
                onChange={(e) => handleGovIdChange('number', e.target.value)}
                placeholder="Enter your ID number"
                className="bg-white text-black placeholder-gray-500" />
            </div>
          </div>
        </>);
      case 4:
        return (<>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="shopLocation" className="text-white">Shop Location</Label>
              <Textarea
                id="shopLocation"
                name="shopLocation"
                value={formData.shopLocation}
                onChange={handleInputChange}
                placeholder="Describe your shop location"
                rows={4}
                className="bg-white text-black placeholder-gray-500" />
            </div>
            <div className="space-y-2">
              <Label htmlFor="image" className="text-white">Upload Image</Label>
              <Input
                id="image"
                type="file"
                onChange={handleImageUpload}
                accept="image/*"
                className="bg-white text-black" />
              {selectedImage && <p className="text-sm text-green-400 mt-1">Image selected: {selectedImage.name}</p>}
            </div>
          </div>
        </>);
      default:
        return null
    }
  }

  return (
    (<div
      className="min-h-screen flex items-center justify-center bg-gradient-to-b from-slate-950 to-indigo-950 p-4 relative overflow-hidden">
      <Card
        className="w-full max-w-lg border-slate-700 bg-slate-800/90 backdrop-blur-sm text-white relative z-10">
        <CardHeader>
          <CardTitle className="text-2xl font-bold text-white">{formPages[currentPage]}</CardTitle>
          <CardDescription className="text-slate-300">
            Step {currentPage + 1} of {formPages.length}
          </CardDescription>
        </CardHeader>
        <CardContent>
          {renderFormPage()}
        </CardContent>
        <CardFooter className="flex justify-between">
          <Button
            onClick={handleBack}
            disabled={currentPage === 0}
            variant="outline"
            className="text-white border-white hover:bg-slate-700">
            <ArrowLeft className="mr-2 h-4 w-4" /> Back
          </Button>
          {currentPage === formPages.length - 1 ? (
            <Button
              onClick={handleSubmit}
              disabled={loading || !selectedImage}
              variant="default"
              className="bg-indigo-600 hover:bg-indigo-700 text-white">
              {loading ? 'Submitting...' : 'Submit'} <Share2 className="ml-2 h-4 w-4" />
            </Button>
          ) : (
            <Button
              onClick={handleNext}
              variant="default"
              className="bg-indigo-600 hover:bg-indigo-700 text-white">
              Next <ArrowRight className="ml-2 h-4 w-4" />
            </Button>
          )}
        </CardFooter>
      </Card>
    </div>)
  );
}
