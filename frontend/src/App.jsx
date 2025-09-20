import { useState } from 'react'
import './App.css'
import Home from './Pages/Home.jsx'
// setting up react router
import { BrowserRouter as Router, Route, Routes } from 'react-router-dom'
import ColorPaletteGenerator from './Components/ColorPaletteGenerator.jsx'

function App() {
  

  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home/>} />
        <Route path="/color" element={<ColorPaletteGenerator/>}/>
      </Routes>
    </Router>
    
  )
}

export default App
