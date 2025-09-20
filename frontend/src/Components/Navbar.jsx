import React from 'react'
import { Link } from 'react-router-dom'

function Navbar() {
  return (
    <div className='z-100'>
        {/* Navbar */}
        <nav className="bg-white shadow-md fixed w-full z-10 top-0 left-0">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="flex items-center justify-between h-16">
                    <div className="flex items-center">
                        <a href="/" className="text-xl font-bold text-indigo-600">ColorGen</a>
                    </div>
                    <div className="hidden md:block">
                        <div className="ml-10 flex items-baseline space-x-4">
                            <Link to="/" className="text-gray-800 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium">Home</Link>
                            <Link to="/color" className="text-gray-800 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium">Color</Link>
                            <Link to="/contact" className="text-gray-800 hover:text-indigo-600 px-3 py-2 rounded-md text-sm font-medium">Contact</Link>
                        </div>
                    </div>
                </div>
            </div>
        </nav>
        {/* Spacer to account for fixed navbar */}
        <div className="h-16"></div>  
    </div>
  )
}

export default Navbar