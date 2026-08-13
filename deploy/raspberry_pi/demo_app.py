import argparse
import base64
import hashlib
import json
import os
import sqlite3
import tempfile
import uuid
import random
from datetime import datetime
from pathlib import Path
from flask import Flask, jsonify, redirect, render_template_string, request, session, url_for
from functools import wraps
from inference import DRPipeline

AUTH_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RetinaAI - Authentication</title>
    
    <!-- Google Fonts: IBM Plex Sans -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Material Symbols Outlined -->
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />

    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'clinical-blue': '#005596',
                        'clinical-green': '#008a4b',
                        'clinical-orange': '#f05a28',
                        'bg-light': '#f8fafc',
                        'card-bg': '#ffffff',
                        'text-primary': '#1e293b',
                        'text-muted': '#64748b',
                        'border-color': '#e2e8f0'
                    },
                    fontFamily: {
                        sans: ['IBM Plex Sans', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    
    <style>
        body {
            font-family: 'IBM Plex Sans', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }
        
        .fade-in {
            animation: fadeIn 0.4s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .tab-btn {
            border-bottom: 2px solid transparent;
            transition: all 0.2s;
        }
        .tab-btn.active {
            border-bottom: 2px solid #005596;
            color: #005596;
            font-weight: 600;
        }

        .pulse-ring {
            animation: pulse-eye 2s infinite ease-out;
            transform-origin: center;
        }
        .scan-line {
            animation: eye-scan 3s infinite linear;
            transform-origin: center;
        }
        @keyframes pulse-eye {
            0% { transform: scale(0.8); opacity: 0.8; }
            100% { transform: scale(1.3); opacity: 0; }
        }
        @keyframes eye-scan {
            0% { transform: translateY(-20px); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(20px); opacity: 0; }
        }
        .retina-logo { transition: transform 0.3s; }
        .retina-logo:hover { transform: scale(1.05); }
    </style>
</head>
<body class="min-h-screen flex m-0">

    <!-- Left Hero Area (Hidden on mobile) -->
    <div class="hidden md:flex flex-col justify-center w-1/2 p-12 text-white" style="background: linear-gradient(135deg, #005596, #003d6b);">
        
        <!-- Eye Logo Inverse Variant -->
        <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" class="retina-logo h-20 w-20 mb-8 self-start text-white">
            <path d="M10,50 Q50,10 90,50 Q50,90 10,50 Z" fill="none" stroke="currentColor" stroke-width="4"/>
            <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" stroke-width="4"/>
            <circle cx="50" cy="50" r="8" fill="currentColor" class="pulse-ring"/>
            <circle cx="50" cy="50" r="8" fill="currentColor"/>
            <line x1="20" y1="50" x2="80" y2="50" stroke="#00ffff" stroke-width="2" class="scan-line" opacity="0.8"/>
        </svg>

        <h1 class="text-5xl font-bold mb-4">RetinaAI</h1>
        <p class="text-xl mb-12 text-blue-100">Edge AI Diabetic Retinopathy Screening</p>
        
        <div class="space-y-6">
            <div class="flex items-center gap-4">
                <span class="material-symbols-outlined text-3xl">psychology</span>
                <span class="text-lg">AI-Powered Screening</span>
            </div>
            <div class="flex items-center gap-4">
                <span class="material-symbols-outlined text-3xl">visibility</span>
                <span class="text-lg">Grad-CAM Explainability</span>
            </div>
            <div class="flex items-center gap-4">
                <span class="material-symbols-outlined text-3xl">router</span>
                <span class="text-lg">Edge Deployment Ready</span>
            </div>
        </div>
    </div>

    <!-- Right Auth Area -->
    <div class="w-full md:w-1/2 flex flex-col justify-center items-center p-6 sm:p-12 fade-in">
        <div class="w-full max-w-md bg-white rounded-xl shadow-lg border border-border-color p-8 relative">
            <div class="flex flex-col items-center mb-8">
                
                <!-- Eye Logo Normal Variant -->
                <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" class="retina-logo h-12 w-12 mb-3 md:hidden text-clinical-blue">
                    <path d="M10,50 Q50,10 90,50 Q50,90 10,50 Z" fill="none" stroke="currentColor" stroke-width="4"/>
                    <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" stroke-width="4"/>
                    <circle cx="50" cy="50" r="8" fill="currentColor" class="pulse-ring"/>
                    <circle cx="50" cy="50" r="8" fill="currentColor"/>
                    <line x1="20" y1="50" x2="80" y2="50" stroke="#00ffff" stroke-width="2" class="scan-line" opacity="0.8"/>
                </svg>

                <h2 class="text-2xl font-bold text-clinical-blue md:hidden">RetinaAI</h2>
                
                <div class="flex w-full mt-6 border-b border-border-color">
                    <button onclick="switchTab('login')" id="tabLogin" class="tab-btn active w-1/2 py-3 text-center text-text-muted hover:text-text-primary focus:outline-none">Sign In</button>
                    <button onclick="switchTab('signup')" id="tabSignup" class="tab-btn w-1/2 py-3 text-center text-text-muted hover:text-text-primary focus:outline-none">Sign Up</button>
                </div>
            </div>

            <!-- Login Form -->
            <form id="loginForm" class="space-y-5">
                <div>
                    <label class="block text-sm font-medium text-text-primary mb-1">Email</label>
                    <input type="email" id="loginEmail" required class="w-full px-4 py-2 border border-border-color rounded-md focus:ring-2 focus:ring-clinical-blue focus:border-clinical-blue outline-none transition-colors">
                </div>
                <div>
                    <label class="block text-sm font-medium text-text-primary mb-1">Password</label>
                    <input type="password" id="loginPassword" required class="w-full px-4 py-2 border border-border-color rounded-md focus:ring-2 focus:ring-clinical-blue focus:border-clinical-blue outline-none transition-colors">
                </div>
                <div id="loginError" class="hidden text-red-600 text-sm font-medium"></div>
                <button type="submit" class="w-full bg-clinical-blue hover:bg-blue-800 text-white py-3 rounded-md font-medium transition-colors flex justify-center items-center gap-2">
                    Sign In
                </button>
            </form>

            <!-- Signup Form -->
            <form id="signupForm" class="space-y-5 hidden">
                <div>
                    <label class="block text-sm font-medium text-text-primary mb-1">Full Name</label>
                    <input type="text" id="signupName" required class="w-full px-4 py-2 border border-border-color rounded-md focus:ring-2 focus:ring-clinical-blue focus:border-clinical-blue outline-none transition-colors">
                </div>
                <div>
                    <label class="block text-sm font-medium text-text-primary mb-1">Email</label>
                    <input type="email" id="signupEmail" required class="w-full px-4 py-2 border border-border-color rounded-md focus:ring-2 focus:ring-clinical-blue focus:border-clinical-blue outline-none transition-colors">
                </div>
                <div>
                    <label class="block text-sm font-medium text-text-primary mb-1">Password</label>
                    <input type="password" id="signupPassword" required minlength="6" class="w-full px-4 py-2 border border-border-color rounded-md focus:ring-2 focus:ring-clinical-blue focus:border-clinical-blue outline-none transition-colors">
                </div>
                <div>
                    <label class="block text-sm font-medium text-text-primary mb-1">Confirm Password</label>
                    <input type="password" id="signupConfirm" required class="w-full px-4 py-2 border border-border-color rounded-md focus:ring-2 focus:ring-clinical-blue focus:border-clinical-blue outline-none transition-colors">
                </div>
                <div id="signupError" class="hidden text-red-600 text-sm font-medium"></div>
                <div id="signupSuccess" class="hidden text-clinical-green text-sm font-medium"></div>
                <button type="submit" class="w-full bg-clinical-blue hover:bg-blue-800 text-white py-3 rounded-md font-medium transition-colors flex justify-center items-center gap-2">
                    Create Account
                </button>
            </form>
        </div>
        
        <p class="mt-8 text-sm text-text-muted italic">For authorized healthcare personnel only</p>
    </div>

    <script>
        function switchTab(tab) {
            document.getElementById('loginForm').classList.add('hidden');
            document.getElementById('signupForm').classList.add('hidden');
            document.getElementById('tabLogin').classList.remove('active');
            document.getElementById('tabSignup').classList.remove('active');
            
            if (tab === 'login') {
                document.getElementById('loginForm').classList.remove('hidden');
                document.getElementById('tabLogin').classList.add('active');
            } else {
                document.getElementById('signupForm').classList.remove('hidden');
                document.getElementById('tabSignup').classList.add('active');
            }
        }

        document.getElementById('loginForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const email = document.getElementById('loginEmail').value;
            const password = document.getElementById('loginPassword').value;
            const errorDiv = document.getElementById('loginError');
            
            try {
                const res = await fetch('/api/login', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({email, password})
                });
                const data = await res.json();
                
                if (data.success) {
                    window.location.href = '/dashboard';
                } else {
                    errorDiv.textContent = data.error || 'Login failed';
                    errorDiv.classList.remove('hidden');
                }
            } catch(err) {
                errorDiv.textContent = 'Server error occurred.';
                errorDiv.classList.remove('hidden');
            }
        });

        document.getElementById('signupForm').addEventListener('submit', async (e) => {
            e.preventDefault();
            const name = document.getElementById('signupName').value;
            const email = document.getElementById('signupEmail').value;
            const password = document.getElementById('signupPassword').value;
            const confirm = document.getElementById('signupConfirm').value;
            
            const errorDiv = document.getElementById('signupError');
            const successDiv = document.getElementById('signupSuccess');
            
            errorDiv.classList.add('hidden');
            successDiv.classList.add('hidden');
            
            if (password !== confirm) {
                errorDiv.textContent = 'Passwords do not match';
                errorDiv.classList.remove('hidden');
                return;
            }
            
            try {
                const res = await fetch('/api/signup', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({name, email, password})
                });
                const data = await res.json();
                
                if (data.success) {
                    successDiv.textContent = 'Account created successfully! Patient ID: ' + data.patient_id + '. Redirecting to login...';
                    successDiv.classList.remove('hidden');
                    setTimeout(() => {
                        switchTab('login');
                        document.getElementById('loginEmail').value = email;
                    }, 2000);
                } else {
                    errorDiv.textContent = data.error || 'Signup failed';
                    errorDiv.classList.remove('hidden');
                }
            } catch(err) {
                errorDiv.textContent = 'Server error occurred.';
                errorDiv.classList.remove('hidden');
            }
        });
    </script>
</body>
</html>"""

DASHBOARD_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RetinaAI - Medical Diagnostic UI</title>
    
    <!-- Google Fonts: IBM Plex Sans -->
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    
    <!-- Material Symbols Outlined -->
    <link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet" />

    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com?plugins=forms,container-queries"></script>
    <script>
        tailwind.config = {
            theme: {
                extend: {
                    colors: {
                        'clinical-blue': '#005596',
                        'clinical-green': '#008a4b',
                        'clinical-orange': '#f05a28',
                        'bg-light': '#f8fafc',
                        'card-bg': '#ffffff',
                        'text-primary': '#1e293b',
                        'text-muted': '#64748b',
                        'border-color': '#e2e8f0'
                    },
                    fontFamily: {
                        sans: ['IBM Plex Sans', 'sans-serif'],
                    }
                }
            }
        }
    </script>
    
    <style>
        body {
            font-family: 'IBM Plex Sans', sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
        }
        
        .material-symbols-outlined {
            font-variation-settings: 'FILL' 1, 'wght' 400, 'GRAD' 0, 'opsz' 24;
            vertical-align: middle;
        }
        
        .loader-overlay {
            display: none;
            position: fixed;
            top: 0; left: 0; right: 0; bottom: 0;
            background: rgba(255, 255, 255, 0.85);
            backdrop-filter: blur(4px);
            z-index: 50;
            place-items: center;
        }
        
        .loader-overlay.active {
            display: grid !important;
        }

        .pulse-ring {
            animation: pulse-eye 2s infinite ease-out;
            transform-origin: center;
        }
        .scan-line {
            animation: eye-scan 3s infinite linear;
            transform-origin: center;
        }
        @keyframes pulse-eye {
            0% { transform: scale(0.8); opacity: 0.8; }
            100% { transform: scale(1.3); opacity: 0; }
        }
        @keyframes eye-scan {
            0% { transform: translateY(-20px); opacity: 0; }
            10% { opacity: 1; }
            90% { opacity: 1; }
            100% { transform: translateY(20px); opacity: 0; }
        }
        .retina-logo { transition: transform 0.3s; }
        .retina-logo:hover { transform: scale(1.05); }

        .anim-pulse-fast {
            animation: pulse-fast 1s infinite alternate;
        }
        @keyframes pulse-fast {
            from { transform: scale(0.95); opacity: 0.8; }
            to { transform: scale(1.05); opacity: 1; }
        }

        .fade-in {
            animation: fadeIn 0.4s ease-in-out;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        .view-btn {
            padding: 0.5rem 1rem;
            font-weight: 500;
            border-radius: 0.375rem;
            transition: all 0.2s;
            color: #64748b;
        }
        .view-btn:hover {
            color: #1e293b;
            background-color: #f1f5f9;
        }
        .view-btn.active {
            background-color: #005596;
            color: white;
        }
        
        /* Top Tabs Navbar */
        .nav-tab {
            border-bottom: 2px solid transparent;
            padding: 1rem;
            color: #64748b;
            font-weight: 500;
            cursor: pointer;
            transition: all 0.2s;
        }
        .nav-tab:hover {
            color: #1e293b;
        }
        .nav-tab.active {
            border-bottom: 2px solid #005596;
            color: #005596;
        }

        /* Stepper pulses */
        .stepper-circle.active {
            animation: pulse-step 1.5s infinite;
        }
        @keyframes pulse-step {
            0% { box-shadow: 0 0 0 0 rgba(0, 85, 150, 0.4); }
            70% { box-shadow: 0 0 0 10px rgba(0, 85, 150, 0); }
            100% { box-shadow: 0 0 0 0 rgba(0, 85, 150, 0); }
        }

        .dropzone {
            border: 2px dashed #cbd5e1;
            transition: border-color 0.2s, background-color 0.2s;
        }
        .dropzone.dragover {
            border-color: #005596;
            background-color: #f0f7ff;
        }
        
        @media print {
            header button, #dropzone-container, #stepper-container, .nav-tab {
                display: none !important;
            }
            body {
                background: white;
            }
            .card-bg {
                border: none;
                box-shadow: none;
            }
        }
    </style>
</head>
<body class="min-h-screen flex flex-col">

    <!-- 1. STICKY HEADER -->
    <header class="sticky top-0 z-40 bg-white border-b border-border-color shadow-sm px-4 sm:px-6 lg:px-8">
        <div class="flex items-center justify-between max-w-7xl mx-auto py-3">
            <div class="flex items-center gap-4">
                <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" class="retina-logo h-8 w-8 text-clinical-blue">
                    <path d="M10,50 Q50,10 90,50 Q50,90 10,50 Z" fill="none" stroke="currentColor" stroke-width="4"/>
                    <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" stroke-width="4"/>
                    <circle cx="50" cy="50" r="8" fill="currentColor" class="pulse-ring"/>
                    <circle cx="50" cy="50" r="8" fill="currentColor"/>
                    <line x1="20" y1="50" x2="80" y2="50" stroke="#00ffff" stroke-width="2" class="scan-line" opacity="0.8"/>
                </svg>
                <h1 class="text-xl font-bold text-clinical-blue hidden sm:block">RetinaAI</h1>
                <div class="w-px h-6 bg-border-color hidden sm:block"></div>
                <div class="flex items-center gap-2">
                    <div class="text-sm font-medium text-text-primary bg-slate-100 px-3 py-1 rounded-full">
                        {{ patient_id }}
                    </div>
                    <span class="hidden sm:inline text-sm text-text-muted">{{ patient_name }}</span>
                </div>
            </div>
            
            <div class="flex items-center gap-3">
                <div class="hidden sm:flex items-center gap-2 mr-4">
                    <div class="w-2.5 h-2.5 rounded-full bg-clinical-green anim-pulse-fast"></div>
                    <span class="text-sm font-medium text-text-muted">System Ready</span>
                </div>
                <button id="btnDownloadReport" onclick="generateReport()" class="hidden bg-clinical-blue hover:bg-blue-800 text-white px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2">
                    <span class="material-symbols-outlined text-[20px]">download</span>
                    Download Report
                </button>
                <button onclick="generateReport()" class="border border-border-color text-text-primary hover:bg-slate-50 px-4 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-2">
                    <span class="material-symbols-outlined text-[20px]">print</span>
                    Print
                </button>
                <a href="/logout" class="border border-red-200 text-red-600 hover:bg-red-50 px-3 py-2 rounded-md text-sm font-medium transition-colors flex items-center gap-1">
                    <span class="material-symbols-outlined text-[18px]">logout</span>
                    Logout
                </a>
            </div>
        </div>

        <div class="flex max-w-7xl mx-auto gap-4">
            <div id="tabNewScreening" class="nav-tab active" onclick="switchNav('new-screening')">New Screening</div>
            <div id="tabHistory" class="nav-tab" onclick="switchNav('history')">Previous Results</div>
        </div>
    </header>

    <main class="flex-grow p-4 sm:p-6 lg:p-8 max-w-7xl mx-auto w-full space-y-6">
        
        <!-- NEW SCREENING VIEW -->
        <div id="screening-view" class="fade-in space-y-6">
            <!-- IMAGE ACQUISITION BAR -->
            <section id="dropzone-container" class="bg-card-bg rounded-lg border border-border-color shadow-sm p-4 sm:p-6">
                <form id="uploadForm" class="flex flex-col md:flex-row gap-6 items-center">
                    <div id="dropzone" class="dropzone flex-grow w-full rounded-lg flex items-center p-6 cursor-pointer relative" onclick="document.getElementById('fileInput').click()">
                        <input type="file" id="fileInput" name="file" accept=".jpg,.jpeg,.png" class="hidden">
                        
                        <div id="uploadPrompt" class="w-full flex flex-col items-center justify-center text-center space-y-2 pointer-events-none">
                            <span class="material-symbols-outlined text-4xl text-text-muted">add_photo_alternate</span>
                            <h3 class="font-medium text-text-primary text-lg">Upload Fundus Image</h3>
                            <p class="text-sm text-text-muted">PNG or JPG, up to 10MB</p>
                        </div>
                        
                        <div id="imagePreviewContainer" class="hidden w-full flex items-center justify-between pointer-events-none">
                            <div class="flex items-center gap-4">
                                <img id="imagePreview" src="" alt="Preview" class="h-16 w-16 object-cover rounded-md border border-border-color">
                                <div class="text-left">
                                    <p id="filenameDisplay" class="font-medium text-text-primary truncate max-w-xs"></p>
                                    <p id="filesizeDisplay" class="text-sm text-text-muted"></p>
                                </div>
                            </div>
                            <span class="material-symbols-outlined text-clinical-blue">check_circle</span>
                        </div>
                    </div>
                    
                    <div class="w-full md:w-auto flex-shrink-0">
                        <button type="submit" id="btnSubmit" class="w-full md:w-auto bg-clinical-blue hover:bg-blue-800 disabled:opacity-50 disabled:cursor-not-allowed text-white px-8 py-3 rounded-md text-base font-medium transition-colors flex items-center justify-center gap-2" disabled>
                            <span class="material-symbols-outlined">play_circle</span>
                            Run Screening
                        </button>
                    </div>
                </form>
            </section>

            <!-- PIPELINE PROGRESS STEPPER -->
            <section id="stepper-container" class="hidden bg-card-bg rounded-lg border border-border-color shadow-sm p-6 fade-in">
                <div class="flex items-center justify-between max-w-3xl mx-auto relative">
                    <!-- Connectors -->
                    <div class="absolute top-5 left-0 w-full flex justify-between z-0 px-[10%]">
                        <div id="conn1" class="h-1 flex-grow bg-slate-200 transition-colors duration-500 mx-2"></div>
                        <div id="conn2" class="h-1 flex-grow bg-slate-200 transition-colors duration-500 mx-2"></div>
                    </div>
                    
                    <!-- Steps -->
                    <div class="relative z-10 flex flex-col items-center gap-2 w-1/3">
                        <div id="step1-circle" class="stepper-circle w-10 h-10 rounded-full flex items-center justify-center bg-slate-200 text-text-muted font-bold transition-all duration-300">
                            <span class="step-num">1</span>
                            <span class="material-symbols-outlined hidden step-icon text-white text-[20px]">check</span>
                        </div>
                        <div class="text-center">
                            <p class="text-sm font-medium text-text-primary">Preprocessing</p>
                            <p id="step1-sub" class="text-xs text-text-muted">(CLAHE)</p>
                        </div>
                    </div>
                    
                    <div class="relative z-10 flex flex-col items-center gap-2 w-1/3">
                        <div id="step2-circle" class="stepper-circle w-10 h-10 rounded-full flex items-center justify-center bg-slate-200 text-text-muted font-bold transition-all duration-300">
                            <span class="step-num">2</span>
                            <span class="material-symbols-outlined hidden step-icon text-white text-[20px]">check</span>
                        </div>
                        <div class="text-center">
                            <p class="text-sm font-medium text-text-primary">Anomaly Gate</p>
                            <p id="step2-sub" class="text-xs text-text-muted">Pending</p>
                        </div>
                    </div>
                    
                    <div class="relative z-10 flex flex-col items-center gap-2 w-1/3">
                        <div id="step3-circle" class="stepper-circle w-10 h-10 rounded-full flex items-center justify-center bg-slate-200 text-text-muted font-bold transition-all duration-300">
                            <span class="step-num">3</span>
                            <span class="material-symbols-outlined hidden step-icon text-white text-[20px]">check</span>
                        </div>
                        <div class="text-center">
                            <p class="text-sm font-medium text-text-primary">Severity Grading</p>
                            <p id="step3-sub" class="text-xs text-text-muted">Pending</p>
                        </div>
                    </div>
                </div>
            </section>

            <!-- Placeholder -->
            <section id="results-placeholder" class="bg-card-bg rounded-lg border border-border-color shadow-sm p-16 flex flex-col items-center justify-center text-center">
                <span class="material-symbols-outlined text-6xl text-slate-300 mb-4">visibility</span>
                <h2 class="text-xl font-medium text-text-primary mb-2">No Image Loaded</h2>
                <p class="text-text-muted max-w-md">Upload a fundus image and run the screening to view the diagnostic results here.</p>
            </section>

            <!-- Active Results Content -->
            <div id="results-content" class="hidden space-y-6 fade-in">
                
                <!-- Alert Banner -->
                <div id="alert-banner" class="bg-white rounded-lg shadow-sm border-l-4 p-4 flex items-start gap-4 relative overflow-hidden">
                    <span id="alert-icon" class="material-symbols-outlined text-3xl">info</span>
                    <div class="flex-grow">
                        <h3 id="alert-title" class="text-lg font-bold text-text-primary mb-1">Result Title</h3>
                        <p id="alert-desc" class="text-text-muted">Result description</p>
                    </div>
                    <div class="text-right relative z-10">
                        <p class="text-xs text-text-muted uppercase tracking-wider font-semibold mb-1">Anomaly Score</p>
                        <p id="alert-score" class="text-xl font-bold text-text-primary">0.00</p>
                    </div>
                    
                    <!-- Decorative Pulsing Eye Badge -->
                    <div id="alert-eye-badge" class="absolute right-32 top-1/2 -translate-y-1/2 opacity-20">
                        <svg viewBox="0 0 100 100" class="h-16 w-16 anim-pulse-fast">
                            <path d="M10,50 Q50,10 90,50 Q50,90 10,50 Z" fill="none" stroke="currentColor" stroke-width="4"/>
                            <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" stroke-width="4"/>
                            <circle cx="50" cy="50" r="8" fill="currentColor"/>
                        </svg>
                    </div>
                </div>

                <!-- Dashboard Grid -->
                <div class="grid grid-cols-1 lg:grid-cols-3 gap-6">
                    
                    <!-- Left Column: Metrics & Risk -->
                    <div class="lg:col-span-1 space-y-6">
                        
                        <!-- 2x2 Metrics -->
                        <div class="grid grid-cols-2 gap-4">
                            <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                                <p class="text-xs text-text-muted uppercase mb-1">Status</p>
                                <p id="metric-status" class="text-lg font-bold uppercase"></p>
                            </div>
                            <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                                <p class="text-xs text-text-muted uppercase mb-1">Severity</p>
                                <p id="metric-severity" class="text-lg font-bold text-text-primary"></p>
                            </div>
                            <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                                <p class="text-xs text-text-muted uppercase mb-1">AI Confidence</p>
                                <p id="metric-confidence" class="text-2xl font-bold text-clinical-blue"></p>
                            </div>
                            <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                                <p class="text-xs text-text-muted uppercase mb-1">Analysis Time</p>
                                <p id="metric-time" class="text-lg font-bold text-text-primary"></p>
                            </div>
                        </div>
                        
                        <!-- Risk Distribution (Only if severity present) -->
                        <div id="risk-card" class="bg-white rounded-lg border border-border-color shadow-sm overflow-hidden hidden">
                            <div class="bg-slate-50 px-4 py-3 border-b border-border-color flex items-center gap-2">
                                <span class="material-symbols-outlined text-text-muted text-[20px]">analytics</span>
                                <h3 class="font-semibold text-text-primary text-sm">Risk Distribution</h3>
                            </div>
                            <div class="p-4 space-y-3" id="risk-bars-container">
                                <!-- Populated dynamically -->
                            </div>
                        </div>

                        <!-- Edge Compute Timing -->
                        <div class="bg-white rounded-lg border border-border-color shadow-sm overflow-hidden">
                            <div class="bg-slate-50 px-4 py-3 border-b border-border-color flex items-center gap-2">
                                <span class="material-symbols-outlined text-text-muted text-[20px]">timer</span>
                                <h3 class="font-semibold text-text-primary text-sm">Edge Compute Timing</h3>
                            </div>
                            <div class="p-4 space-y-2 text-sm" id="timing-container">
                                <!-- Populated dynamically -->
                            </div>
                        </div>
                    </div>

                    <!-- Right Column: Visualizations -->
                    <div id="visual-column" class="lg:col-span-2 hidden bg-white rounded-lg border border-border-color shadow-sm flex flex-col">
                        
                        <!-- Toolbar -->
                        <div class="flex items-center justify-between p-4 border-b border-border-color">
                            <div class="flex bg-slate-100 p-1 rounded-lg gap-1" id="view-tabs">
                                <button class="view-btn active text-sm" onclick="switchView('side-by-side', this)">Side-by-Side</button>
                                <button class="view-btn text-sm" onclick="switchView('original', this)">Original</button>
                                <button class="view-btn text-sm" onclick="switchView('heatmap', this)">Heatmap</button>
                            </div>
                            <div class="bg-red-50 text-clinical-orange border border-red-100 px-3 py-1 rounded-full text-xs font-bold flex items-center gap-1">
                                <span class="material-symbols-outlined text-[16px]">healing</span>
                                <span id="badge-lesion-load"></span> Lesion Area
                            </div>
                        </div>

                        <!-- Image Container -->
                        <div class="p-4 bg-slate-50 flex-grow flex items-center justify-center min-h-[400px]">
                            
                            <!-- Side by side view -->
                            <div id="view-side-by-side" class="w-full grid grid-cols-1 md:grid-cols-2 gap-4">
                                <div class="relative w-full aspect-square rounded-lg overflow-hidden border border-border-color shadow-inner bg-black">
                                    <div id="img-orig-sbs" class="absolute inset-0 bg-contain bg-center bg-no-repeat"></div>
                                    <div class="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">Original</div>
                                </div>
                                <div class="relative w-full aspect-square rounded-lg overflow-hidden border border-border-color shadow-inner bg-black">
                                    <div id="img-heat-sbs" class="absolute inset-0 bg-contain bg-center bg-no-repeat"></div>
                                    <div class="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">Grad-CAM Heatmap</div>
                                </div>
                            </div>

                            <!-- Single Views -->
                            <div id="view-original" class="hidden w-full max-w-2xl aspect-square relative rounded-lg overflow-hidden border border-border-color shadow-inner bg-black">
                                <div id="img-orig-single" class="absolute inset-0 bg-contain bg-center bg-no-repeat"></div>
                                <div class="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">Original</div>
                            </div>

                            <div id="view-heatmap" class="hidden w-full max-w-2xl aspect-square relative rounded-lg overflow-hidden border border-border-color shadow-inner bg-black">
                                <div id="img-heat-single" class="absolute inset-0 bg-contain bg-center bg-no-repeat"></div>
                                <div class="absolute top-2 left-2 bg-black/60 text-white text-xs px-2 py-1 rounded backdrop-blur-sm">Grad-CAM Heatmap</div>
                            </div>
                        </div>

                        <!-- Clinical Findings -->
                        <div class="p-4 border-t border-border-color bg-slate-50">
                            <div class="flex gap-3 text-sm">
                                <span class="material-symbols-outlined text-clinical-blue mt-0.5">info</span>
                                <div>
                                    <p class="font-medium text-text-primary mb-1">Clinical Findings</p>
                                    <p class="text-text-muted mb-2"><span id="finding-grade" class="font-semibold"></span>. <span id="finding-lesion"></span></p>
                                    <p class="italic text-xs text-slate-400">Disclaimer: This is an AI-assisted screening tool. Results should be verified by a qualified ophthalmologist.</p>
                                </div>
                            </div>
                        </div>
                    </div>

                </div>
            </div>
        </div>

        <!-- HISTORY TIMELINE VIEW -->
        <div id="history-view" class="hidden fade-in space-y-6">
            <!-- Summary Metrics Row -->
            <div class="grid grid-cols-2 lg:grid-cols-4 gap-4">
                <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                    <p class="text-xs text-text-muted uppercase mb-1">Total Screenings</p>
                    <p id="hist-total" class="text-2xl font-bold text-text-primary">0</p>
                </div>
                <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                    <p class="text-xs text-text-muted uppercase mb-1">Healthy Screenings</p>
                    <p id="hist-healthy" class="text-2xl font-bold text-clinical-green">0</p>
                </div>
                <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                    <p class="text-xs text-text-muted uppercase mb-1">Flagged Screenings</p>
                    <p id="hist-flagged" class="text-2xl font-bold text-clinical-orange">0</p>
                </div>
                <div class="bg-white p-4 rounded-lg border border-border-color shadow-sm">
                    <p class="text-xs text-text-muted uppercase mb-1">Latest Screening</p>
                    <p id="hist-latest-date" class="text-xl font-bold text-text-primary">-</p>
                </div>
            </div>

            <div class="bg-white rounded-lg border border-border-color shadow-sm">
                <div class="bg-slate-50 px-4 py-3 border-b border-border-color flex items-center justify-between">
                    <h3 class="font-semibold text-text-primary">Diagnostic Timeline</h3>
                    <button onclick="fetchHistory()" class="text-clinical-blue hover:text-blue-800 text-sm font-medium flex items-center gap-1">
                        <span class="material-symbols-outlined text-[18px]">refresh</span> Refresh
                    </button>
                </div>
                
                <div id="history-feed" class="p-6 space-y-6">
                    <div class="text-center text-text-muted py-8">Loading timeline...</div>
                </div>
            </div>
        </div>
    </main>

    <!-- LOADING OVERLAY -->
    <div id="loader" class="loader-overlay">
        <div class="bg-white p-8 rounded-xl shadow-xl flex flex-col items-center max-w-sm w-full mx-4 border border-border-color text-center space-y-4">
            
            <svg viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg" class="h-16 w-16 text-clinical-blue mb-2">
                <path d="M10,50 Q50,10 90,50 Q50,90 10,50 Z" fill="none" stroke="currentColor" stroke-width="4"/>
                <circle cx="50" cy="50" r="20" fill="none" stroke="currentColor" stroke-width="4"/>
                <circle cx="50" cy="50" r="8" fill="currentColor" class="pulse-ring"/>
                <circle cx="50" cy="50" r="8" fill="currentColor"/>
                <line x1="20" y1="50" x2="80" y2="50" stroke="#00ffff" stroke-width="2" class="scan-line" opacity="0.8"/>
            </svg>
            
            <h3 class="text-lg font-bold text-clinical-blue">Executing Edge Inference...</h3>
            <p id="loader-text" class="text-sm font-medium text-text-muted transition-opacity">Initializing pipeline...</p>
            <div class="w-full bg-slate-100 h-2 rounded-full mt-2 overflow-hidden">
                <div class="bg-clinical-blue h-full w-1/3 animate-pulse rounded-full"></div>
            </div>
        </div>
    </div>

    <!-- Modals etc... can be expanded if needed -->

    <!-- Scripts -->
    <script>
        const PATIENT_ID = '{{ patient_id }}';
        const PATIENT_NAME = '{{ patient_name }}';
        
        let uploadedImageB64 = null;
        let lastResults = null;
        let historyData = [];

        function switchNav(tab) {
            document.querySelectorAll('.nav-tab').forEach(el => el.classList.remove('active'));
            document.getElementById('screening-view').classList.add('hidden');
            document.getElementById('history-view').classList.add('hidden');
            
            if (tab === 'new-screening') {
                document.getElementById('tabNewScreening').classList.add('active');
                document.getElementById('screening-view').classList.remove('hidden');
            } else if (tab === 'history') {
                document.getElementById('tabHistory').classList.add('active');
                document.getElementById('history-view').classList.remove('hidden');
                fetchHistory();
            }
        }

        async function fetchHistory() {
            try {
                const res = await fetch('/api/history');
                const data = await res.json();
                if (data.success) {
                    historyData = data.history;
                    renderHistory();
                }
            } catch (err) {
                console.error(err);
            }
        }

        function renderHistory() {
            const feed = document.getElementById('history-feed');
            
            if (!historyData || historyData.length === 0) {
                feed.innerHTML = '<div class="text-center text-text-muted py-8">No previous screenings found.</div>';
                document.getElementById('hist-total').textContent = '0';
                document.getElementById('hist-healthy').textContent = '0';
                document.getElementById('hist-flagged').textContent = '0';
                document.getElementById('hist-latest-date').textContent = '-';
                return;
            }

            let normalCount = 0;
            let flaggedCount = 0;
            let html = '';

            historyData.forEach((item, index) => {
                const isNormal = item.gate === 'normal_gate';
                if(isNormal) normalCount++; else flaggedCount++;
                
                const dateObj = new Date(item.timestamp);
                const dateStr = dateObj.toLocaleDateString() + ' ' + dateObj.toLocaleTimeString();

                let imgsHtml = '';
                if(item.original_b64) {
                    const imgSrc = item.original_b64.startsWith('data:') ? item.original_b64 : 'data:image/jpeg;base64,' + item.original_b64;
                    const heatSrc = item.heatmap_b64 ? 'data:image/png;base64,' + item.heatmap_b64 : '';
                    imgsHtml = `
                    <div class="flex gap-4 mt-4">
                        <img src="${imgSrc}" class="h-24 w-24 object-cover rounded border border-border-color bg-black">
                        ${heatSrc ? `<img src="${heatSrc}" class="h-24 w-24 object-cover rounded border border-border-color bg-black">` : ''}
                    </div>
                    `;
                }

                html += `
                <div class="border border-border-color rounded-lg overflow-hidden">
                    <div class="bg-slate-50 px-4 py-3 border-b border-border-color flex justify-between items-center">
                        <div class="flex items-center gap-3">
                            <div class="bg-white border border-border-color px-2 py-1 rounded text-xs font-semibold text-text-muted">
                                ${dateStr}
                            </div>
                            <span class="px-2 py-1 rounded-full text-xs font-bold ${isNormal ? 'bg-green-100 text-clinical-green' : 'bg-orange-100 text-clinical-orange'}">
                                ${isNormal ? 'Normal — No Referral' : 'Flagged — ' + (item.severity_name || 'Refer')}
                            </span>
                        </div>
                    </div>
                    <div class="p-4">
                        <div class="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
                            <div>
                                <p class="text-xs text-text-muted">Anomaly Score</p>
                                <p class="font-medium">${item.anomaly_score.toFixed(4)}</p>
                            </div>
                            <div>
                                <p class="text-xs text-text-muted">Severity Grade</p>
                                <p class="font-medium">${item.severity_name || 'N/A'}</p>
                            </div>
                            <div>
                                <p class="text-xs text-text-muted">Confidence</p>
                                <p class="font-medium">${item.confidence_pct ? item.confidence_pct.toFixed(1)+'%' : 'N/A'}</p>
                            </div>
                            <div>
                                <p class="text-xs text-text-muted">Lesion Area</p>
                                <p class="font-medium">${item.lesion_load ? (item.lesion_load*100).toFixed(1)+'%' : 'N/A'}</p>
                            </div>
                        </div>
                        
                        ${imgsHtml}
                        
                        <div class="mt-4 flex gap-3">
                            <button onclick='generateReportFromHistory(${index})' class="text-sm text-clinical-blue hover:underline flex items-center gap-1">
                                <span class="material-symbols-outlined text-[16px]">download</span> Download Report
                            </button>
                        </div>
                    </div>
                </div>
                `;
            });

            feed.innerHTML = html;
            
            document.getElementById('hist-total').textContent = historyData.length;
            document.getElementById('hist-healthy').textContent = normalCount;
            document.getElementById('hist-flagged').textContent = flaggedCount;
            
            const latest = new Date(historyData[0].timestamp);
            document.getElementById('hist-latest-date').textContent = latest.toLocaleDateString();
        }

        function generateReportFromHistory(index) {
            const item = historyData[index];
            const r = item.full_json;
            const isNormal = r.gate === 'normal_gate';
            const dateStr = new Date(item.timestamp).toLocaleString();
            
            const origB64 = item.original_b64.startsWith('data:') ? item.original_b64 : 'data:image/jpeg;base64,' + item.original_b64;

            let heatmapSection = '';
            if(r.cam_heatmap) {
                heatmapSection = `
                <div class="section" style="page-break-inside: avoid;">
                    <h2>Visual Analysis (Grad-CAM)</h2>
                    <div style="display: flex; gap: 20px;">
                        <div style="flex: 1;">
                            <h4>Original Image</h4>
                            <img src="${origB64}" style="width: 100%; border: 1px solid #ccc;">
                        </div>
                        <div style="flex: 1;">
                            <h4>Heatmap Analysis</h4>
                            <img src="data:image/png;base64,${r.cam_heatmap}" style="width: 100%; border: 1px solid #ccc;">
                        </div>
                    </div>
                    <p style="margin-top: 10px;"><strong>Lesion Load:</strong> ${(r.cam_lesion_load*100).toFixed(1)}%</p>
                </div>`;
            }
            
            let severitySection = '';
            if(r.severity) {
                let probsHtml = '';
                for(let [k,v] of Object.entries(r.severity.probabilities)) {
                    probsHtml += `<li>${k}: ${(v*100).toFixed(2)}%</li>`;
                }
                
                severitySection = `
                <div class="section">
                    <h2>Severity Assessment</h2>
                    <p><strong>Predicted Grade:</strong> ${r.severity.grade_name} (Grade ${r.severity.grade})</p>
                    <p><strong>AI Confidence:</strong> ${r.severity.confidence_pct.toFixed(2)}%</p>
                    <h4>Risk Distribution:</h4>
                    <ul>${probsHtml}</ul>
                </div>`;
            }

            const htmlContent = `
<!DOCTYPE html>
<html>
<head>
    <title>Diagnostic Report - ${PATIENT_NAME} (${PATIENT_ID})</title>
    <link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'IBM Plex Sans', sans-serif; color: #1e293b; max-width: 800px; margin: 0 auto; padding: 40px; background: #fff; line-height: 1.6; }
        .header { border-bottom: 2px solid #005596; padding-bottom: 20px; margin-bottom: 30px; }
        .header h1 { color: #005596; margin: 0 0 10px 0; }
        .meta { display: flex; justify-content: space-between; color: #64748b; font-size: 0.9em; }
        .section { background: #f8fafc; border: 1px solid #e2e8f0; padding: 20px; border-radius: 8px; margin-bottom: 20px; }
        h2 { color: #005596; margin-top: 0; font-size: 1.2em; border-bottom: 1px solid #cbd5e1; padding-bottom: 8px; }
        .alert { padding: 15px; border-left: 5px solid ${isNormal ? '#008a4b' : '#f05a28'}; background: #fff; margin-bottom: 20px; border-radius: 4px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
        .footer { margin-top: 50px; font-size: 0.8em; color: #94a3b8; text-align: center; border-top: 1px solid #e2e8f0; padding-top: 20px; font-style: italic; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RetinaAI Diagnostic Report</h1>
        <div class="meta">
            <span><strong>Patient:</strong> ${PATIENT_NAME} (${PATIENT_ID})</span>
            <span><strong>Date:</strong> ${dateStr}</span>
        </div>
    </div>
    
    <div class="alert">
        <h3 style="margin:0 0 10px 0; color: ${isNormal ? '#008a4b' : '#f05a28'};">
            ${isNormal ? 'Screening Complete: No Anomalies Detected' : 'Referral Recommended: Anomalies Detected'}
        </h3>
        <p style="margin:0;"><strong>Action:</strong> ${r.action}</p>
        <p style="margin:5px 0 0 0;"><strong>Anomaly Score:</strong> ${r.anomaly_score.toFixed(4)}</p>
    </div>
    
    ${severitySection}
    ${heatmapSection}
    
    <div class="section">
        <h2>System Timings</h2>
        <ul style="margin:0; padding-left: 20px;">
            <li>Preprocessing: ${r.timings.preprocess_ms} ms</li>
            <li>Stage 1 (Gate): ${r.timings.stage1_ms} ms</li>
            ${r.timings.stage2_ms ? `<li>Stage 2 (Classify): ${r.timings.stage2_ms} ms</li>` : ''}
            <li><strong>Total Time: ${r.timings.total_ms} ms</strong></li>
        </ul>
    </div>
    
    <div class="footer">
        Disclaimer: This report is generated by an AI-assisted screening tool. The results, interpretations, and visualizations are for informational purposes only and must be verified by a qualified ophthalmologist. This is not a final medical diagnosis.
    </div>
</body>
</html>`;

            const blob = new Blob([htmlContent], { type: 'text/html' });
            const url = URL.createObjectURL(blob);
            
            const timestamp = new Date().toISOString().replace(/[:T]/g, '-').split('.')[0];
            const filename = `RetinaAI_Report_${timestamp}.html`;
            
            const a = document.createElement('a');
            a.href = url;
            a.download = filename;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            URL.revokeObjectURL(url);
        }

        // --- Rest of original scripts for file drop and analysis ---
        const fileInput = document.getElementById('fileInput');
        const dropzone = document.getElementById('dropzone');
        
        // Drag and drop setup
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            if(dropzone) dropzone.addEventListener(eventName, preventDefaults, false);
        });
        function preventDefaults(e) {
            e.preventDefault();
            e.stopPropagation();
        }
        
        ['dragenter', 'dragover'].forEach(eventName => {
            if(dropzone) dropzone.addEventListener(eventName, () => dropzone.classList.add('dragover'), false);
        });
        
        ['dragleave', 'drop'].forEach(eventName => {
            if(dropzone) dropzone.addEventListener(eventName, () => dropzone.classList.remove('dragover'), false);
        });
        
        if(dropzone) dropzone.addEventListener('drop', (e) => {
            let dt = e.dataTransfer;
            let files = dt.files;
            if(files.length > 0) {
                fileInput.files = files;
                handleFileSelect();
            }
        });
        
        if(fileInput) fileInput.addEventListener('change', handleFileSelect);
        
        function handleFileSelect() {
            if(fileInput.files && fileInput.files[0]) {
                const file = fileInput.files[0];
                const reader = new FileReader();
                
                reader.onload = function(e) {
                    uploadedImageB64 = e.target.result;
                    document.getElementById('uploadPrompt').classList.add('hidden');
                    document.getElementById('imagePreviewContainer').classList.remove('hidden');
                    document.getElementById('imagePreview').src = uploadedImageB64;
                    document.getElementById('filenameDisplay').textContent = file.name;
                    document.getElementById('filesizeDisplay').textContent = (file.size / (1024*1024)).toFixed(2) + ' MB';
                    document.getElementById('btnSubmit').disabled = false;
                }
                reader.readAsDataURL(file);
            }
        }

        if(document.getElementById('uploadForm')) {
            document.getElementById('uploadForm').addEventListener('submit', async (e) => {
                e.preventDefault();
                if(!fileInput.files[0]) return;
                
                const loader = document.getElementById('loader');
                const loaderText = document.getElementById('loader-text');
                const stepperContainer = document.getElementById('stepper-container');
                const resultsPlaceholder = document.getElementById('results-placeholder');
                const resultsContent = document.getElementById('results-content');
                
                loader.classList.add('active');
                resultsPlaceholder.classList.add('hidden');
                resultsContent.classList.add('hidden');
                stepperContainer.classList.remove('hidden');
                document.getElementById('btnDownloadReport').classList.add('hidden');
                
                resetStepper();
                
                loaderText.textContent = "Stage 1: Anomaly Gate Evaluation";
                setStepperState(1, 'active');
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                
                try {
                    setTimeout(() => {
                        loaderText.textContent = "Stage 2: Severity Classification & Grad-CAM";
                        setStepperState(1, 'completed');
                        setStepperState(2, 'active');
                    }, 1000);

                    const response = await fetch('/api/analyze', {
                        method: 'POST',
                        body: formData
                    });
                    
                    if(!response.ok) throw new Error("API request failed");
                    const data = await response.json();
                    lastResults = data;
                    
                    setTimeout(() => {
                        loader.classList.remove('active');
                        renderResults(data);
                    }, 1500);
                    
                } catch(error) {
                    console.error("Error analyzing image:", error);
                    alert("An error occurred during analysis.");
                    loader.classList.remove('active');
                }
            });
        }

        function resetStepper() {
            const steps = [1,2,3];
            steps.forEach(s => {
                const circle = document.getElementById(`step${s}-circle`);
                circle.className = "stepper-circle w-10 h-10 rounded-full flex items-center justify-center bg-slate-200 text-text-muted font-bold transition-all duration-300";
                circle.querySelector('.step-num').classList.remove('hidden');
                circle.querySelector('.step-icon').classList.add('hidden');
                circle.querySelector('.step-icon').textContent = 'check';
                if(s==2) document.getElementById('step2-sub').textContent = 'Pending';
                if(s==3) document.getElementById('step3-sub').textContent = 'Pending';
            });
            document.getElementById('conn1').className = "h-1 flex-grow bg-slate-200 transition-colors duration-500 mx-2";
            document.getElementById('conn2').className = "h-1 flex-grow bg-slate-200 transition-colors duration-500 mx-2";
        }
        
        function setStepperState(stepNum, state, extra=null) {
            const circle = document.getElementById(`step${stepNum}-circle`);
            const num = circle.querySelector('.step-num');
            const icon = circle.querySelector('.step-icon');
            
            circle.classList.remove('bg-slate-200', 'bg-clinical-blue', 'bg-clinical-green', 'bg-clinical-orange', 'text-text-muted', 'text-white', 'active');
            
            if(state === 'active') {
                circle.classList.add('bg-clinical-blue', 'text-white', 'active');
                num.classList.remove('hidden');
                icon.classList.add('hidden');
            } else if (state === 'completed') {
                circle.classList.add('bg-clinical-green', 'text-white');
                num.classList.add('hidden');
                icon.classList.remove('hidden');
                icon.textContent = 'check';
                
                if(stepNum < 3) {
                    document.getElementById(`conn${stepNum}`).classList.replace('bg-slate-200', 'bg-clinical-green');
                }
            } else if (state === 'flagged') {
                circle.classList.add('bg-clinical-orange', 'text-white');
                num.classList.add('hidden');
                icon.classList.remove('hidden');
                icon.textContent = 'priority_high';
                if(stepNum < 3) {
                    document.getElementById(`conn${stepNum}`).classList.replace('bg-slate-200', 'bg-clinical-orange');
                }
            }
        }

        function renderResults(res) {
            const content = document.getElementById('results-content');
            content.classList.remove('hidden');
            document.getElementById('btnDownloadReport').classList.remove('hidden');
            
            const isNormal = res.gate === 'normal_gate';
            
            setStepperState(1, 'completed');
            if(isNormal) {
                setStepperState(2, 'completed');
                document.getElementById('step2-sub').textContent = 'Passed';
            } else {
                setStepperState(2, 'flagged');
                document.getElementById('step2-sub').textContent = 'Flagged';
                setStepperState(3, 'completed');
                document.getElementById('step3-sub').textContent = res.severity.grade_name;
            }

            const banner = document.getElementById('alert-banner');
            const title = document.getElementById('alert-title');
            const desc = document.getElementById('alert-desc');
            const icon = document.getElementById('alert-icon');
            const eyeBadge = document.getElementById('alert-eye-badge');
            
            banner.className = `bg-white rounded-lg shadow-sm border-l-4 p-4 flex items-start gap-4 relative overflow-hidden ${isNormal ? 'border-clinical-green' : 'border-clinical-orange'}`;
            icon.className = `material-symbols-outlined text-3xl ${isNormal ? 'text-clinical-green' : 'text-clinical-orange'}`;
            eyeBadge.className = `absolute right-32 top-1/2 -translate-y-1/2 opacity-20 ${isNormal ? 'text-clinical-green' : 'text-clinical-orange'}`;
            
            icon.textContent = isNormal ? 'check_circle' : 'warning';
            title.textContent = isNormal ? 'Screening Complete: No Anomalies Detected' : 'Referral Recommended: Anomalies Detected';
            desc.textContent = res.action;
            document.getElementById('alert-score').textContent = res.anomaly_score.toFixed(4);
            
            const statusEl = document.getElementById('metric-status');
            statusEl.textContent = isNormal ? 'Normal' : 'Flagged';
            statusEl.className = `text-lg font-bold uppercase ${isNormal ? 'text-clinical-green' : 'text-clinical-orange'}`;
            
            document.getElementById('metric-severity').textContent = res.severity ? res.severity.grade_name : 'N/A';
            document.getElementById('metric-confidence').textContent = res.severity ? res.severity.confidence_pct.toFixed(1) + '%' : 'N/A';
            document.getElementById('metric-time').textContent = res.timings.total_ms + ' ms';
            
            const riskCard = document.getElementById('risk-card');
            const riskBars = document.getElementById('risk-bars-container');
            if(res.severity && res.severity.probabilities) {
                riskCard.classList.remove('hidden');
                riskBars.innerHTML = '';
                
                const probs = res.severity.probabilities;
                const colors = ['bg-clinical-green', 'bg-blue-300', 'bg-clinical-blue', 'bg-clinical-orange', 'bg-red-800'];
                const names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative'];
                const pKeys = Object.keys(probs);
                
                names.forEach((name, idx) => {
                    const keyMatch = pKeys.find(k => k.includes(name) || name.includes(k));
                    const val = keyMatch ? probs[keyMatch] : 0;
                    const pct = (val * 100).toFixed(1);
                    const isPredicted = idx === res.severity.grade;
                    
                    riskBars.innerHTML += `
                        <div class="flex items-center gap-2 ${isPredicted ? 'font-black text-text-primary' : 'text-sm text-text-muted'}">
                            <div class="w-24 flex-shrink-0 text-right truncate">${name}</div>
                            <div class="flex-grow h-3 bg-slate-100 rounded-full overflow-hidden">
                                <div class="h-full ${colors[idx]}" style="width: ${pct}%"></div>
                            </div>
                            <div class="w-12 text-right">${pct}%</div>
                        </div>
                    `;
                });
            } else {
                riskCard.classList.add('hidden');
            }

            const tCont = document.getElementById('timing-container');
            tCont.innerHTML = `
                <div class="flex justify-between border-b border-slate-100 pb-1"><span>Preprocessing:</span> <span class="font-medium">${res.timings.preprocess_ms} ms</span></div>
                <div class="flex justify-between border-b border-slate-100 pb-1 pt-1"><span>Stage 1 (Gate):</span> <span class="font-medium">${res.timings.stage1_ms} ms</span></div>
                ${res.timings.stage2_ms ? `<div class="flex justify-between border-b border-slate-100 pb-1 pt-1"><span>Stage 2 (Classify):</span> <span class="font-medium">${res.timings.stage2_ms} ms</span></div>` : ''}
            `;
            
            const visCol = document.getElementById('visual-column');
            if(res.severity && res.cam_heatmap) {
                visCol.classList.remove('hidden');
                
                document.getElementById('badge-lesion-load').textContent = (res.cam_lesion_load * 100).toFixed(1) + '%';
                
                const heatmapDataUrl = 'data:image/png;base64,' + res.cam_heatmap;
                
                document.getElementById('img-orig-sbs').style.backgroundImage = `url('${uploadedImageB64}')`;
                document.getElementById('img-heat-sbs').style.backgroundImage = `url('${heatmapDataUrl}')`;
                
                document.getElementById('img-orig-single').style.backgroundImage = `url('${uploadedImageB64}')`;
                document.getElementById('img-heat-single').style.backgroundImage = `url('${heatmapDataUrl}')`;
                
                document.getElementById('finding-grade').textContent = `Predicted: ${res.severity.grade_name}`;
                document.getElementById('finding-lesion').textContent = `Grad-CAM analysis highlights areas contributing to the classification. Lesion load metric is ${(res.cam_lesion_load*100).toFixed(1)}%.`;
                
                switchView('side-by-side', document.querySelector('.view-btn'));
            } else {
                visCol.classList.add('hidden');
            }
            
            setTimeout(() => {
                content.scrollIntoView({ behavior: 'smooth', block: 'start' });
            }, 100);
        }
        
        function switchView(mode, btn) {
            document.querySelectorAll('.view-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            
            document.getElementById('view-side-by-side').classList.add('hidden');
            document.getElementById('view-original').classList.add('hidden');
            document.getElementById('view-heatmap').classList.add('hidden');
            
            document.getElementById(`view-${mode}`).classList.remove('hidden');
        }

        function generateReport() {
            if(!lastResults) return;
            // Fake entry mapping to use our combined history generator
            const dummyHistoryItem = {
                timestamp: new Date().toISOString(),
                gate: lastResults.gate,
                action: lastResults.action,
                anomaly_score: lastResults.anomaly_score,
                severity_grade: lastResults.severity ? lastResults.severity.grade : null,
                severity_name: lastResults.severity ? lastResults.severity.grade_name : null,
                confidence_pct: lastResults.severity ? lastResults.severity.confidence_pct : null,
                lesion_load: lastResults.cam_lesion_load,
                original_b64: uploadedImageB64,
                heatmap_b64: lastResults.cam_heatmap,
                full_json: lastResults
            };
            
            // Push dummy to historyData so generator works
            historyData.push(dummyHistoryItem);
            generateReportFromHistory(historyData.length - 1);
            historyData.pop();
        }
    </script>
</body>
</html>
"""

def init_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS patients (
            id TEXT PRIMARY KEY,
            patient_id TEXT UNIQUE NOT NULL,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            diabetes_type TEXT DEFAULT 'Not Specified',
            created_at TEXT NOT NULL
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS screening_history (
            id TEXT PRIMARY KEY,
            patient_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            gate TEXT NOT NULL,
            action TEXT NOT NULL,
            anomaly_score REAL NOT NULL,
            severity_grade INTEGER,
            severity_name TEXT,
            confidence_pct REAL,
            lesion_load REAL,
            blood_sugar_level REAL,
            diabetes_type TEXT,
            rejected INTEGER DEFAULT 0,
            rejection_reason TEXT,
            original_b64 TEXT,
            heatmap_b64 TEXT,
            full_json TEXT NOT NULL,
            FOREIGN KEY(patient_id) REFERENCES patients(patient_id)
        )
    """)
    # Migration: add new columns to existing tables if they don't exist
    try:
        c.execute("ALTER TABLE patients ADD COLUMN diabetes_type TEXT DEFAULT 'Not Specified'")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE screening_history ADD COLUMN blood_sugar_level REAL")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE screening_history ADD COLUMN diabetes_type TEXT")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE screening_history ADD COLUMN rejected INTEGER DEFAULT 0")
    except sqlite3.OperationalError:
        pass
    try:
        c.execute("ALTER TABLE screening_history ADD COLUMN rejection_reason TEXT")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

def hash_password(password):
    salt = "retinaAI_salt_"
    return hashlib.sha256((salt + password).encode('utf-8')).hexdigest()

def verify_password(password, password_hash):
    return hash_password(password) == password_hash

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not session.get('patient_id'):
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated_function

def create_app(models_dir, threads):
    app = Flask(__name__)
    app.secret_key = os.urandom(24).hex()
    
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'patients.db')
    init_db(db_path)
    
    pipe = DRPipeline(models_dir=models_dir, threads=threads)
    
    def get_db():
        return sqlite3.connect(db_path)

    @app.route('/', methods=['GET'])
    def index():
        if session.get('patient_id'):
            return redirect(url_for('dashboard'))
        return redirect(url_for('login'))

    @app.route('/login', methods=['GET'])
    def login():
        if session.get('patient_id'):
            return redirect(url_for('dashboard'))
        return render_template_string(AUTH_TEMPLATE)

    @app.route('/api/login', methods=['POST'])
    def api_login():
        data = request.json
        email = data.get('email')
        password = data.get('password')
        
        conn = get_db()
        c = conn.cursor()
        c.execute('SELECT id, patient_id, name, password_hash FROM patients WHERE email = ?', (email,))
        row = c.fetchone()
        conn.close()
        
        if row and verify_password(password, row[3]):
            session['patient_id'] = row[1]
            session['patient_name'] = row[2]
            session['patient_email'] = email
            return jsonify({"success": True, "name": row[2], "patient_id": row[1]})
        
        return jsonify({"success": False, "error": "Invalid email or password"})

    @app.route('/api/signup', methods=['POST'])
    def api_signup():
        data = request.json
        name = data.get('name')
        email = data.get('email')
        password = data.get('password')
        diabetes_type = data.get('diabetes_type', 'Not Specified')
        
        if not name or not email or not password:
            return jsonify({"success": False, "error": "All fields are required"})
        
        if diabetes_type not in ['Prediabetic', 'Type 1 Diabetes', 'Type 2 Diabetes', 'Not Specified']:
            diabetes_type = 'Not Specified'
            
        conn = get_db()
        c = conn.cursor()
        
        c.execute('SELECT id FROM patients WHERE email = ?', (email,))
        if c.fetchone():
            conn.close()
            return jsonify({"success": False, "error": "Email already registered"})
            
        patient_id = f"IND-{random.randint(1000, 9999)}-PT"
        
        c.execute('SELECT id FROM patients WHERE patient_id = ?', (patient_id,))
        while c.fetchone():
            patient_id = f"IND-{random.randint(1000, 9999)}-PT"
            c.execute('SELECT id FROM patients WHERE patient_id = ?', (patient_id,))
            
        db_id = str(uuid.uuid4())
        hashed = hash_password(password)
        created_at = datetime.now().isoformat()
        
        c.execute('''
            INSERT INTO patients (id, patient_id, name, email, password_hash, diabetes_type, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (db_id, patient_id, name, email, hashed, diabetes_type, created_at))
        
        conn.commit()
        conn.close()
        
        session['diabetes_type'] = diabetes_type
        return jsonify({"success": True, "name": name, "patient_id": patient_id, "diabetes_type": diabetes_type})

    @app.route('/dashboard', methods=['GET'])
    @login_required
    def dashboard():
        return render_template_string(
            DASHBOARD_TEMPLATE,
            patient_name=session.get('patient_name'),
            patient_id=session.get('patient_id'),
            patient_email=session.get('patient_email')
        )

    @app.route('/api/analyze', methods=['POST'])
    @login_required
    def analyze():
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
        
        # Task 4: Blood sugar level (required field)
        blood_sugar = request.form.get('blood_sugar', None)
        if blood_sugar:
            try:
                blood_sugar = float(blood_sugar)
            except ValueError:
                blood_sugar = None
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_file:
            file.save(temp_file.name)
            tmp_path = temp_file.name
            
        try:
            with open(tmp_path, "rb") as f:
                uploaded_b64 = base64.b64encode(f.read()).decode('utf-8')
            
            results = pipe.run(tmp_path)
            
            record_id = str(uuid.uuid4())
            ts = datetime.now().isoformat()
            gate = results.get('gate', '')
            action = results.get('action', '')
            anomaly_score = results.get('anomaly_score', 0.0)
            
            severity = results.get('severity') or {}
            severity_grade = severity.get('grade')
            severity_name = severity.get('grade_name')
            confidence_pct = severity.get('confidence_pct')
            
            lesion_load = results.get('cam_lesion_load')
            heatmap_b64 = results.get('cam_heatmap')
            
            # Task 2: Rejection info
            rejection = severity.get('rejection', {})
            rejected = 1 if rejection.get('rejected', False) else 0
            rejection_reason = rejection.get('reason', '')
            
            # Task 4: Get diabetes type from patient record
            conn = get_db()
            c = conn.cursor()
            c.execute('SELECT diabetes_type FROM patients WHERE patient_id = ?', (session['patient_id'],))
            pt_row = c.fetchone()
            diabetes_type = pt_row[0] if pt_row else 'Not Specified'
            
            # Task 1: Fetch previous screening for comparison
            c.execute('''
                SELECT severity_grade, severity_name, confidence_pct, lesion_load,
                       timestamp, blood_sugar_level, rejected
                FROM screening_history
                WHERE patient_id = ? AND severity_grade IS NOT NULL
                ORDER BY timestamp DESC LIMIT 1
            ''', (session['patient_id'],))
            prev_row = c.fetchone()
            
            comparison = None
            if prev_row and severity_grade is not None:
                prev_grade = prev_row[0]
                prev_name = prev_row[1]
                prev_conf = prev_row[2]
                prev_lesion = prev_row[3]
                prev_date = prev_row[4]
                prev_sugar = prev_row[5]
                
                if prev_grade is not None:
                    grade_diff = severity_grade - prev_grade
                    if grade_diff > 0:
                        grade_change = "worsened"
                    elif grade_diff < 0:
                        grade_change = "improved"
                    else:
                        grade_change = "stable"
                    
                    lesion_diff = None
                    if lesion_load is not None and prev_lesion is not None:
                        lesion_diff = round(lesion_load - prev_lesion, 4)
                    
                    comparison = {
                        "previous_grade": prev_grade,
                        "previous_name": prev_name,
                        "current_grade": severity_grade,
                        "current_name": severity_name,
                        "grade_change": grade_change,
                        "grade_diff": grade_diff,
                        "previous_lesion_load": prev_lesion,
                        "current_lesion_load": lesion_load,
                        "lesion_load_diff": lesion_diff,
                        "previous_confidence": prev_conf,
                        "previous_date": prev_date,
                        "previous_blood_sugar": prev_sugar,
                    }
            
            # Store to database
            c.execute('''
                INSERT INTO screening_history (
                    id, patient_id, timestamp, gate, action, anomaly_score,
                    severity_grade, severity_name, confidence_pct, lesion_load,
                    blood_sugar_level, diabetes_type, rejected, rejection_reason,
                    original_b64, heatmap_b64, full_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                record_id, session['patient_id'], ts, gate, action, anomaly_score,
                severity_grade, severity_name, confidence_pct, lesion_load,
                blood_sugar, diabetes_type, rejected, rejection_reason,
                uploaded_b64, heatmap_b64, json.dumps(results)
            ))
            conn.commit()
            conn.close()
            
            results['history_id'] = record_id
            results['blood_sugar_level'] = blood_sugar
            results['diabetes_type'] = diabetes_type
            if comparison:
                results['comparison'] = comparison
            
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
        return jsonify(results)

    @app.route('/api/history', methods=['GET'])
    @login_required
    def history():
        conn = get_db()
        c = conn.cursor()
        c.execute('''
            SELECT 
                id, timestamp, gate, action, anomaly_score, 
                severity_grade, severity_name, confidence_pct, lesion_load, 
                original_b64, heatmap_b64, full_json
            FROM screening_history
            WHERE patient_id = ?
            ORDER BY timestamp DESC
        ''', (session['patient_id'],))
        
        rows = c.fetchall()
        conn.close()
        
        history_list = []
        for row in rows:
            history_list.append({
                "id": row[0],
                "timestamp": row[1],
                "gate": row[2],
                "action": row[3],
                "anomaly_score": row[4],
                "severity_grade": row[5],
                "severity_name": row[6],
                "confidence_pct": row[7],
                "lesion_load": row[8],
                "original_b64": row[9],
                "heatmap_b64": row[10],
                "full_json": json.loads(row[11])
            })
            
        return jsonify({"success": True, "history": history_list})

    @app.route('/health', methods=['GET'])
    def health():
        return jsonify({"status": "ok"})
        
    @app.route('/logout', methods=['GET'])
    def logout():
        session.clear()
        return redirect(url_for('login'))

    return app

def main():
    parser = argparse.ArgumentParser(description="RetinaAI Demo App")
    parser.add_argument('--models', type=str, default='models', help="Path to models directory")
    parser.add_argument('--host', type=str, default='0.0.0.0', help="Host address")
    parser.add_argument('--port', type=int, default=5000, help="Port number")
    parser.add_argument('--threads', type=int, default=1, help="Number of threads for inference")
    args = parser.parse_args()
    
    app = create_app(args.models, args.threads)
    app.run(host=args.host, port=args.port, debug=False)

if __name__ == "__main__":
    main()
