import React, { useState, useRef } from 'react';
import { Palette, Image, Type, Copy, Download, Sparkles, Upload, X } from 'lucide-react';
import Navbar from './Navbar';

export default function ColorPaletteGenerator() {
  const [activeTab, setActiveTab] = useState('text');
  const [textPrompt, setTextPrompt] = useState('');
  const [uploadedImage, setUploadedImage] = useState(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generatedPalette, setGeneratedPalette] = useState([]);
  const [copiedIndex, setCopiedIndex] = useState(-1);
  const [error, setError] = useState(null);
  const fileInputRef = useRef(null);
  
  // Backend API endpoint - adjust the URL based on your FastAPI server
  const API_BASE_URL = 'http://localhost:8000'; // Change this to match your backend URL

  // API call to FastAPI backend
  const callColorGenerationAPI = async (data, isImage = false) => {
    const endpoint = isImage ? '/generate-from-image' : '/generate-from-text';
    
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      method: 'POST',
      headers: {
        'Content-Type': isImage ? 'multipart/form-data' : 'application/json',
      },
      body: isImage ? data : JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`API call failed: ${response.status} ${response.statusText}`);
    }

    return await response.json();
  };

  const handleGenerate = async () => {
    if (!textPrompt.trim() && !uploadedImage) return;
    
    setIsGenerating(true);
    setError(null);
    
    try {
      let result;
      
      if (activeTab === 'text' && textPrompt.trim()) {
        // Send text prompt to backend
        result = await callColorGenerationAPI({ prompt: textPrompt });
      } else if (activeTab === 'image' && uploadedImage) {
        // Send image to backend
        const formData = new FormData();
        // Convert data URL to blob
        const response = await fetch(uploadedImage);
        const blob = await response.blob();
        formData.append('image', blob, 'uploaded-image.jpg');
        
        result = await callColorGenerationAPI(formData, true);
      }
      
      // Expected response format: { colors: [{ hex: '#FF6B6B', rgb: 'rgb(255,107,107)', hsl: 'hsl(0,100%,71%)' }, ...] }
      if (result && result.colors) {
        setGeneratedPalette(result.colors);
      } else {
        throw new Error('Invalid response format from API');
      }
    } catch (error) {
      console.error('Error generating palette:', error);
      setError(error.message || 'Failed to generate palette. Please try again.');
    } finally {
      setIsGenerating(false);
    }
  };

  const handleImageUpload = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => setUploadedImage(e.target.result);
      reader.readAsDataURL(file);
    }
  };

  const copyToClipboard = (colorValue, index) => {
    navigator.clipboard.writeText(colorValue);
    setCopiedIndex(index);
    setTimeout(() => setCopiedIndex(-1), 1000);
  };

  const removeImage = () => {
    setUploadedImage(null);
    fileInputRef.current.value = '';
  };

  return (
    <>
    <Navbar/>
    <div className="min-h-screen  p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="text-center pt-5 mb-12">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="p-3 bg-white rounded-2xl shadow-lg">
              <Palette className="w-8 h-8 text-indigo-600" />
            </div>
            <h1 className="text-4xl font-bold text-gray-800">AI Palette</h1>
          </div>
          <p className="text-gray-600 text-lg">Generate beautiful color palettes from text descriptions or images</p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-3xl shadow-xl p-8 mb-8">
          {/* Tab Navigation */}
          <div className="flex bg-gray-100 rounded-2xl p-1 mb-8">
            <button
              onClick={() => setActiveTab('text')}
              className={`flex-1 flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'text'
                  ? 'bg-white shadow-md text-indigo-600 transform scale-[1.02]'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <Type className="w-5 h-5" />
              Text Prompt
            </button>
            <button
              onClick={() => setActiveTab('image')}
              className={`flex-1 flex items-center justify-center gap-2 py-3 px-6 rounded-xl font-medium transition-all duration-300 ${
                activeTab === 'image'
                  ? 'bg-white shadow-md text-indigo-600 transform scale-[1.02]'
                  : 'text-gray-500 hover:text-gray-700'
              }`}
            >
              <Image className="w-5 h-5" />
              Image Upload
            </button>
          </div>

          {/* Text Input */}
          {activeTab === 'text' && (
            <div className="space-y-4">
              <textarea
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                placeholder="Describe your desired palette... (e.g., 'sunset over ocean', 'cozy autumn morning', 'cyberpunk neon')"
                className="w-full h-32 p-4 border-2 border-gray-200 rounded-2xl focus:border-indigo-500 focus:outline-none resize-none text-gray-700 transition-colors duration-200"
              />
            </div>
          )}

          {/* Image Upload */}
          {activeTab === 'image' && (
            <div className="space-y-4">
              {!uploadedImage ? (
                <div
                  onClick={() => fileInputRef.current?.click()}
                  className="border-2 border-dashed border-gray-300 rounded-2xl p-12 text-center cursor-pointer hover:border-indigo-400 hover:bg-indigo-50/50 transition-all duration-300"
                >
                  <Upload className="w-12 h-12 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 text-lg mb-2">Click to upload an image</p>
                  <p className="text-gray-400">JPG, PNG up to 10MB</p>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="image/*"
                    onChange={handleImageUpload}
                    className="hidden"
                  />
                </div>
              ) : (
                <div className="relative rounded-2xl overflow-hidden">
                  <img src={uploadedImage} alt="Uploaded" className="w-full h-64 object-cover" />
                  <button
                    onClick={removeImage}
                    className="absolute top-4 right-4 p-2 bg-black/50 text-white rounded-full hover:bg-black/70 transition-colors duration-200"
                  >
                    <X className="w-5 h-5" />
                  </button>
                </div>
              )}
            </div>
          )}

          {/* Generate Button */}
          <button
            onClick={handleGenerate}
            disabled={isGenerating || (!textPrompt.trim() && !uploadedImage)}
            className="w-full mt-8 py-4 px-8 bg-gradient-to-r from-indigo-600 to-purple-600 text-white rounded-2xl font-semibold text-lg disabled:opacity-50 disabled:cursor-not-allowed hover:from-indigo-700 hover:to-purple-700 transform hover:scale-[1.02] transition-all duration-200 flex items-center justify-center gap-3"
          >
            {isGenerating ? (
              <>
                <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                Generating Magic...
              </>
            ) : (
              <>
                <Sparkles className="w-5 h-5" />
                Generate Palette
              </>
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="bg-red-50 border border-red-200 rounded-2xl p-4 mb-8">
            <div className="flex items-center gap-2">
              <div className="w-5 h-5 bg-red-500 rounded-full flex items-center justify-center">
                <span className="text-white text-xs font-bold">!</span>
              </div>
              <p className="text-red-700 font-medium">Error: {error}</p>
            </div>
          </div>
        )}

        {/* Generated Palette */}
        {generatedPalette.length > 0 && (
          <div className="bg-white rounded-3xl shadow-xl p-8 animate-in slide-in-from-bottom-4 duration-700">
            <div className="flex items-center justify-between mb-6">
              <h2 className="text-2xl font-bold text-gray-800">Your Palette</h2>
              <button className="flex items-center gap-2 px-4 py-2 text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-xl transition-colors duration-200">
                <Download className="w-5 h-5" />
                Export
              </button>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {generatedPalette.map((colorObj, index) => {
                // Handle both old format (string) and new format (object with hex, rgb, hsl)
                const color = typeof colorObj === 'string' ? colorObj : colorObj.hex || colorObj.color;
                const hex = typeof colorObj === 'string' ? colorObj : colorObj.hex || colorObj.color;
                const rgb = typeof colorObj === 'string' ? null : colorObj.rgb;
                const hsl = typeof colorObj === 'string' ? null : colorObj.hsl;

                return (
                  <div
                    key={index}
                    className="group animate-in zoom-in-0 duration-500"
                    style={{ animationDelay: `${index * 100}ms` }}
                  >
                    <div
                      className="h-32 rounded-2xl shadow-lg group-hover:shadow-xl transform group-hover:scale-105 transition-all duration-300 mb-4"
                      style={{ backgroundColor: color }}
                    />
                    
                    {/* Color Information */}
                    <div className="space-y-2">
                      {/* HEX */}
                      <div 
                        className="cursor-pointer p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200"
                        onClick={() => copyToClipboard(hex, `hex-${index}`)}
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-medium text-gray-500 uppercase">HEX</span>
                          {copiedIndex === `hex-${index}` && <span className="text-green-600 text-xs font-medium">Copied!</span>}
                        </div>
                        <p className="font-mono text-sm text-gray-700 mt-1">{hex}</p>
                      </div>

                      {/* RGB */}
                      {rgb && (
                        <div 
                          className="cursor-pointer p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200"
                          onClick={() => copyToClipboard(rgb, `rgb-${index}`)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-medium text-gray-500 uppercase">RGB</span>
                            {copiedIndex === `rgb-${index}` && <span className="text-green-600 text-xs font-medium">Copied!</span>}
                          </div>
                          <p className="font-mono text-sm text-gray-700 mt-1">{rgb}</p>
                        </div>
                      )}

                      {/* HSL */}
                      {hsl && (
                        <div 
                          className="cursor-pointer p-2 bg-gray-50 rounded-lg hover:bg-gray-100 transition-colors duration-200"
                          onClick={() => copyToClipboard(hsl, `hsl-${index}`)}
                        >
                          <div className="flex items-center justify-between">
                            <span className="text-xs font-medium text-gray-500 uppercase">HSL</span>
                            {copiedIndex === `hsl-${index}` && <span className="text-green-600 text-xs font-medium">Copied!</span>}
                          </div>
                          <p className="font-mono text-sm text-gray-700 mt-1">{hsl}</p>
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Palette Info */}
            <div className="mt-8 p-4 bg-gray-50 rounded-2xl">
              <div className="flex items-center justify-between text-sm text-gray-600">
                <span>Generated from: {activeTab === 'text' ? `"${textPrompt}"` : 'Uploaded image'}</span>
                <span>{generatedPalette.length} colors</span>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
    </>
  );
}