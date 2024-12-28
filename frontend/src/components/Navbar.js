// src/components/Navbar.js
import React from 'react';
import './Navbar.css';
import { BookOpen, HelpCircle, Info, Moon, Sun } from 'lucide-react';
import { Analytics } from '@mui/icons-material';
import { useTheme } from '../context/ThemeContext';

function Navbar() {
  const { isDarkMode, toggleTheme } = useTheme();

  return (
    <nav className="navbar">
      <div className="navbar-container">
        <div className="navbar-brand">
          <Analytics className="brand-icon" />
          <h1 className="brand-title">ML Dashboard</h1>
        </div>
        <ul className="navbar-links">
          <li>
            <button onClick={toggleTheme} className="theme-toggle">
              {isDarkMode ? <Sun /> : <Moon />}
              <span>{isDarkMode ? 'Light' : 'Dark'}</span>
            </button>
          </li>
          <li>
            <a href="/docs">
              <BookOpen />
              <span>Docs</span>
            </a>
          </li>
          <li>
            <a href="/help">
              <HelpCircle />
              <span>Help</span>
            </a>
          </li>
          <li>
            <a href="/about">
              <Info />
              <span>About</span>
            </a>
          </li>
        </ul>
      </div>
    </nav>
  );
}

export default Navbar;
