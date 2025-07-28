// Main JavaScript for LLM Risk Visualizer Website

document.addEventListener('DOMContentLoaded', function() {
    // Initialize all components
    initNavigation();
    initDemoTabs();
    initModal();
    initAnimations();
    initScrollEffects();
});

// Navigation functionality
function initNavigation() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    const navbar = document.querySelector('.navbar');

    // Mobile menu toggle
    if (hamburger && navMenu) {
        hamburger.addEventListener('click', () => {
            hamburger.classList.toggle('active');
            navMenu.classList.toggle('active');
        });

        // Close mobile menu when clicking on nav links
        document.querySelectorAll('.nav-menu a').forEach(link => {
            link.addEventListener('click', () => {
                hamburger.classList.remove('active');
                navMenu.classList.remove('active');
            });
        });
    }

    // Navbar scroll effect
    let lastScroll = 0;
    window.addEventListener('scroll', () => {
        const currentScroll = window.pageYOffset;
        
        if (currentScroll <= 0) {
            navbar.classList.remove('scroll-up');
            return;
        }
        
        if (currentScroll > lastScroll && !navbar.classList.contains('scroll-down')) {
            // Scroll down
            navbar.classList.remove('scroll-up');
            navbar.classList.add('scroll-down');
        } else if (currentScroll < lastScroll && navbar.classList.contains('scroll-down')) {
            // Scroll up
            navbar.classList.remove('scroll-down');
            navbar.classList.add('scroll-up');
        }
        
        lastScroll = currentScroll;
    });

    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                const headerOffset = 80;
                const elementPosition = target.getBoundingClientRect().top;
                const offsetPosition = elementPosition + window.pageYOffset - headerOffset;

                window.scrollTo({
                    top: offsetPosition,
                    behavior: 'smooth'
                });
            }
        });
    });
}

// Demo tabs functionality
function initDemoTabs() {
    const demoTabs = document.querySelectorAll('.demo-tab');
    const demoPanels = document.querySelectorAll('.demo-panel');

    demoTabs.forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and panels
            demoTabs.forEach(t => t.classList.remove('active'));
            demoPanels.forEach(p => p.classList.remove('active'));

            // Add active class to clicked tab
            tab.classList.add('active');

            // Show corresponding panel
            const targetDemo = tab.getAttribute('data-demo');
            const targetPanel = document.getElementById(`demo-${targetDemo}`);
            if (targetPanel) {
                targetPanel.classList.add('active');
            }
        });
    });
}

// Modal functionality
function initModal() {
    const modal = document.getElementById('demoModal');
    const modalClose = document.querySelector('.modal-close');
    const modalTitle = document.getElementById('modalTitle');
    const modalVideo = document.getElementById('modalVideo');

    // Close modal when clicking close button
    if (modalClose) {
        modalClose.addEventListener('click', () => {
            closeModal();
        });
    }

    // Close modal when clicking outside of it
    window.addEventListener('click', (event) => {
        if (event.target === modal) {
            closeModal();
        }
    });

    // Close modal with escape key
    document.addEventListener('keydown', (event) => {
        if (event.key === 'Escape' && modal.style.display === 'block') {
            closeModal();
        }
    });

    function closeModal() {
        if (modal) {
            modal.style.display = 'none';
            document.body.style.overflow = 'auto';
        }
    }
}

// Play demo functionality
function playDemo(demoType) {
    const modal = document.getElementById('demoModal');
    const modalTitle = document.getElementById('modalTitle');
    const modalVideo = document.getElementById('modalVideo');

    const demoInfo = {
        'overview': {
            title: 'Platform Overview Demo',
            description: 'Complete walkthrough of all platform features'
        },
        'ai-detection': {
            title: 'AI Risk Detection Demo',
            description: 'Real-time AI-powered risk analysis'
        },
        'ar-vr': {
            title: 'AR/VR Visualization Demo',
            description: 'Immersive 3D data exploration'
        },
        'blockchain': {
            title: 'Blockchain Audit Demo',
            description: 'Immutable audit trail system'
        }
    };

    if (modal && modalTitle && modalVideo) {
        const info = demoInfo[demoType] || demoInfo['overview'];
        modalTitle.textContent = info.title;
        
        // In a real implementation, you would embed actual video content here
        modalVideo.innerHTML = `
            <div class="video-placeholder-large">
                <i class="fas fa-play-circle"></i>
                <h3>${info.title}</h3>
                <p>${info.description}</p>
                <p class="video-note">Note: This is a demonstration website. In a real deployment, actual demo videos would be embedded here using platforms like YouTube, Vimeo, or direct video files.</p>
                <div style="margin-top: 2rem;">
                    <a href="https://github.com/WolfgangDremmler/LLM-Risk-Visualizer" class="btn btn-primary">
                        <i class="fab fa-github"></i> View Source Code
                    </a>
                </div>
            </div>
        `;

        modal.style.display = 'block';
        document.body.style.overflow = 'hidden';
    }
}

// Animations and scroll effects
function initAnimations() {
    // Animate statistics in hero section
    const stats = document.querySelectorAll('.stat-number');
    const animateStats = () => {
        stats.forEach(stat => {
            const target = parseInt(stat.textContent.replace(/\D/g, ''));
            let current = 0;
            const increment = target / 50;
            const timer = setInterval(() => {
                current += increment;
                if (current >= target) {
                    current = target;
                    clearInterval(timer);
                }
                
                // Format the number based on original content
                const originalText = stat.textContent;
                if (originalText.includes('K')) {
                    stat.textContent = Math.floor(current) + 'K+';
                } else if (originalText.includes('%')) {
                    stat.textContent = Math.floor(current) + '%';
                } else {
                    stat.textContent = Math.floor(current);
                }
            }, 50);
        });
    };

    // Trigger animation when hero section is visible
    const heroSection = document.querySelector('.hero');
    if (heroSection) {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    animateStats();
                    observer.unobserve(entry.target);
                }
            });
        });
        observer.observe(heroSection);
    }

    // Animate chart bars in hero dashboard
    const chartBars = document.querySelectorAll('.bar');
    chartBars.forEach((bar, index) => {
        setTimeout(() => {
            bar.style.opacity = '0';
            bar.style.height = '0';
            setTimeout(() => {
                bar.style.transition = 'all 0.8s ease';
                bar.style.opacity = '1';
                bar.style.height = bar.dataset.height || bar.style.height;
            }, 100);
        }, index * 200);
    });
}

// Scroll effects for sections
function initScrollEffects() {
    // Fade in animation for feature cards
    const observerOptions = {
        threshold: 0.1,
        rootMargin: '0px 0px -50px 0px'
    };

    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.style.opacity = '1';
                entry.target.style.transform = 'translateY(0)';
                observer.unobserve(entry.target);
            }
        });
    }, observerOptions);

    // Apply animation to feature cards
    document.querySelectorAll('.feature-card, .doc-card, .contact-card').forEach(card => {
        card.style.opacity = '0';
        card.style.transform = 'translateY(30px)';
        card.style.transition = 'opacity 0.6s ease, transform 0.6s ease';
        observer.observe(card);
    });

    // Parallax effect for hero background
    window.addEventListener('scroll', () => {
        const scrolled = window.pageYOffset;
        const hero = document.querySelector('.hero');
        if (hero) {
            const rate = scrolled * -0.5;
            hero.style.transform = `translateY(${rate}px)`;
        }
    });

    // Progress indicator for long pages
    const progressBar = createProgressBar();
    window.addEventListener('scroll', updateProgressBar);

    function createProgressBar() {
        const progress = document.createElement('div');
        progress.id = 'scroll-progress';
        progress.style.cssText = `
            position: fixed;
            top: 0;
            left: 0;
            width: 0%;
            height: 3px;
            background: linear-gradient(90deg, #667eea, #764ba2);
            z-index: 10000;
            transition: width 0.1s ease;
        `;
        document.body.appendChild(progress);
        return progress;
    }

    function updateProgressBar() {
        const progress = document.getElementById('scroll-progress');
        if (progress) {
            const scrollTop = window.pageYOffset;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = (scrollTop / docHeight) * 100;
            progress.style.width = scrollPercent + '%';
        }
    }
}

// Utility functions
function debounce(func, wait, immediate) {
    let timeout;
    return function executedFunction() {
        const context = this;
        const args = arguments;
        const later = function() {
            timeout = null;
            if (!immediate) func.apply(context, args);
        };
        const callNow = immediate && !timeout;
        clearTimeout(timeout);
        timeout = setTimeout(later, wait);
        if (callNow) func.apply(context, args);
    };
}

// Enhanced mobile navigation
function initMobileNavigation() {
    const hamburger = document.querySelector('.hamburger');
    const navMenu = document.querySelector('.nav-menu');
    
    if (hamburger && navMenu) {
        // Create mobile menu styles
        const style = document.createElement('style');
        style.textContent = `
            @media (max-width: 768px) {
                .nav-menu {
                    position: fixed;
                    left: -100%;
                    top: 70px;
                    flex-direction: column;
                    background-color: white;
                    width: 100%;
                    text-align: center;
                    transition: 0.3s;
                    box-shadow: 0 10px 27px rgba(0, 0, 0, 0.05);
                    padding: 2rem 0;
                }

                .nav-menu.active {
                    left: 0;
                }

                .nav-menu li {
                    margin: 1rem 0;
                }

                .hamburger.active span:nth-child(2) {
                    opacity: 0;
                }

                .hamburger.active span:nth-child(1) {
                    transform: translateY(8px) rotate(45deg);
                }

                .hamburger.active span:nth-child(3) {
                    transform: translateY(-8px) rotate(-45deg);
                }
            }
        `;
        document.head.appendChild(style);
    }
}

// Initialize enhanced mobile navigation
initMobileNavigation();

// Feature card hover effects
document.querySelectorAll('.feature-card').forEach(card => {
    card.addEventListener('mouseenter', function() {
        this.style.transform = 'translateY(-10px) scale(1.02)';
    });

    card.addEventListener('mouseleave', function() {
        this.style.transform = 'translateY(0) scale(1)';
    });
});

// Interactive demo buttons
document.querySelectorAll('.demo-btn').forEach(btn => {
    btn.addEventListener('click', function() {
        // Add click animation
        this.style.transform = 'scale(0.95)';
        setTimeout(() => {
            this.style.transform = 'scale(1)';
        }, 150);
    });
});

// Copy to clipboard functionality for code examples
function copyToClipboard(text) {
    navigator.clipboard.writeText(text).then(function() {
        // Show success message
        const message = document.createElement('div');
        message.textContent = 'Copied to clipboard!';
        message.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            background: #10b981;
            color: white;
            padding: 12px 20px;
            border-radius: 8px;
            z-index: 10000;
            animation: slideInRight 0.3s ease;
        `;
        document.body.appendChild(message);
        
        setTimeout(() => {
            message.remove();
        }, 3000);
    });
}

// Add CSS animations
const animationStyles = document.createElement('style');
animationStyles.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }

    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    .fade-in-up {
        animation: fadeInUp 0.6s ease forwards;
    }
`;
document.head.appendChild(animationStyles);

// Error handling for missing elements
function safeQuerySelector(selector, callback) {
    const element = document.querySelector(selector);
    if (element && callback) {
        callback(element);
    }
    return element;
}

// Performance optimization - lazy load images
function initLazyLoading() {
    const images = document.querySelectorAll('img[data-src]');
    const imageObserver = new IntersectionObserver((entries, observer) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                const img = entry.target;
                img.src = img.dataset.src;
                img.classList.remove('lazy');
                imageObserver.unobserve(img);
            }
        });
    });

    images.forEach(img => imageObserver.observe(img));
}

// Initialize lazy loading if there are lazy images
if (document.querySelectorAll('img[data-src]').length > 0) {
    initLazyLoading();
}

// Console welcome message for developers
console.log(`
%c🚀 LLM Risk Visualizer
%cWelcome to the developer console!

This is an advanced AI-powered risk management platform.
- 🤖 AI Risk Detection
- ⛓️ Blockchain Audit Trails  
- 🔒 Federated Learning
- 🥽 AR/VR Visualization
- ⚛️ Quantum-Safe Security

GitHub: https://github.com/WolfgangDremmler/LLM-Risk-Visualizer
Email: dremmlerwolfgang559@gmail.com

%cFeel free to explore the code and contribute!`,
'color: #667eea; font-size: 24px; font-weight: bold;',
'color: #64748b; font-size: 14px; line-height: 1.5;',
'color: #10b981; font-size: 12px; font-weight: bold;'
);

// Export functions for external use
window.LLMRiskVisualizer = {
    playDemo,
    copyToClipboard,
    safeQuerySelector
};