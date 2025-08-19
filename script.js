document.addEventListener('DOMContentLoaded', () => {
    const loginForm = document.getElementById('loginForm');
    const loginContainer = document.getElementById('loginContainer');
    const dashboardWrapper = document.getElementById('dashboardWrapper');
    const loginMessage = document.getElementById('loginMessage');
    const welcomeMessage = document.getElementById('welcomeMessage');
    const logoutBtn = document.getElementById('logoutBtn');
    const navLinks = document.querySelectorAll('.sidebar-nav a');
    const contentSections = document.querySelectorAll('.content-section');
    const adminOnlyElements = document.querySelectorAll('.admin-only'); // Elements only for admin
    const contactForm = document.getElementById('contactForm');
    const statusPesan = document.getElementById('statusPesan');

    // --- Simulated User Data (Insecure for Production) ---
    const users = {
        'admin': { password: 'adminpassword', role: 'admin', name: 'Admin Dashboard' },
        'pegawai1': { password: 'pegawaipassword', role: 'pegawai', name: 'Pegawai John Doe' }
    };

    let currentUserRole = null; // To store current logged-in user's role

    // --- Functions for UI Management ---

    // Show Login Screen
    const showLogin = () => {
        loginContainer.classList.remove('hidden');
        dashboardWrapper.classList.add('hidden');
        loginForm.reset(); // Clear form fields
        loginMessage.textContent = ''; // Clear any previous messages
        currentUserRole = null; // Clear role on logout
        localStorage.removeItem('loggedInUser'); // Clear stored user info
    };

    // Show Dashboard
    const showDashboard = (username, role) => {
        loginContainer.classList.add('hidden');
        dashboardWrapper.classList.remove('hidden');
        welcomeMessage.textContent = `Selamat Datang, ${users[username].name}!`;
        currentUserRole = role;
        updateDashboardVisibility(role); // Adjust visibility based on role
        showSection('overview'); // Default to overview
        setActiveLink(document.querySelector('.sidebar-nav a[href="#overview"]'));
    };

    // Update Dashboard Elements Based on Role
    const updateDashboardVisibility = (role) => {
        adminOnlyElements.forEach(el => {
            if (role === 'admin') {
                el.style.display = 'block'; // Or 'flex', 'grid', etc., based on original display
            } else {
                el.style.display = 'none';
            }
        });

        // If a non-admin user somehow lands on an admin-only section, redirect them
        const currentSectionId = window.location.hash.substring(1);
        if (role !== 'admin' && document.getElementById(currentSectionId)?.classList.contains('admin-only')) {
            showSection('overview');
            setActiveLink(document.querySelector('.sidebar-nav a[href="#overview"]'));
            history.pushState(null, '', '#overview');
        }
    };

    // Function to show a specific section
    const showSection = (id) => {
        contentSections.forEach(section => {
            if (section.id === id) {
                section.classList.add('active');
            } else {
                section.classList.remove('active');
            }
        });
    };

    // Function to set active link in sidebar
    const setActiveLink = (linkElement) => {
        navLinks.forEach(link => link.classList.remove('active'));
        if (linkElement) { // Check if linkElement exists
            linkElement.classList.add('active');
        }
    };

    // --- Event Listeners ---

    // Handle Login Form Submission
    if (loginForm) {
        loginForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const usernameInput = document.getElementById('username').value.trim();
            const passwordInput = document.getElementById('password').value.trim();

            if (users[usernameInput] && users[usernameInput].password === passwordInput) {
                // Successful login
                localStorage.setItem('loggedInUser', JSON.stringify({ username: usernameInput, role: users[usernameInput].role }));
                showDashboard(usernameInput, users[usernameInput].role);
            } else {
                // Failed login
                loginMessage.textContent = 'Username atau password salah!';
            }
        });
    }

    // Handle Logout Button
    if (logoutBtn) {
        logoutBtn.addEventListener('click', () => {
            showLogin();
        });
    }

    // Handle Navigation Clicks
    navLinks.forEach(link => {
        link.addEventListener('click', function(e) {
            e.preventDefault();
            const targetId = this.getAttribute('href').substring(1);

            // Prevent non-admin users from accessing admin-only sections via direct click
            if (currentUserRole !== 'admin' && document.getElementById(targetId)?.classList.contains('admin-only')) {
                alert('Anda tidak memiliki akses ke halaman ini.'); // Or a more graceful message
                return;
            }

            showSection(targetId);
            setActiveLink(this);
            history.pushState(null, '', `#${targetId}`);
        });
    });

    // Handle Contact Form Submission (existing functionality)
    if (contactForm) {
        contactForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const nama = document.getElementById('nama').value;
            statusPesan.textContent = `Terima kasih, ${nama}! Pesan Anda telah dikirim.`;
            statusPesan.style.display = 'block';
            this.reset();
            setTimeout(() => {
                statusPesan.style.display = 'none';
                statusPesan.textContent = '';
            }, 5000);
        });
    }

    // --- Initial Load Check ---
    // Check if user is already logged in (e.g., from a previous session)
    const storedUser = localStorage.getItem('loggedInUser');
    if (storedUser) {
        try {
            const user = JSON.parse(storedUser);
            if (users[user.username] && users[user.username].role === user.role) {
                showDashboard(user.username, user.role);
            } else {
                showLogin(); // Data in localStorage is invalid
            }
        } catch (e) {
            console.error("Failed to parse stored user data:", e);
            showLogin();
        }
    } else {
        showLogin(); // No user stored, show login screen
    }
});