// src/components/Footer.js
import React from 'react';
import './Footer.css';
import { Github } from 'lucide-react';

function Footer() {
    return (
        <footer className="footer">
            <div className="footer-content">
                <p className="footer-text">
                    💪 Built by Sajjad 👽 and Reza 😎
                </p>
                <a 
                    href="https://github.com/rht789/MachineLearningProject" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="github-link"
                >
                    <Github size={20} />
                    View on GitHub
                </a>
            </div>
        </footer>
    );
}

export default Footer;
