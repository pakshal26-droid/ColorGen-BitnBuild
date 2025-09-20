import React from 'react'
import ColorPaletteGenerator from '../Components/ColorPaletteGenerator'
import Navbar from '../Components/Navbar'

function Home() {
  return (
    <div className=''>
        <Navbar/>
        <div className='flex flex-col h-[80vh] justify-center items-center '>
            <h1>
                Color Palette Generator
            </h1>
        </div>
    </div>
  )
}

export default Home