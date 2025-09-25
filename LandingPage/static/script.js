// --- Profile Dropdown Logic ---
function toggleProfileDropdown() {
    const dropdown = document.getElementById('profileDropdown');
    dropdown.classList.toggle('show');
}

// Hide dropdown when clicking outside
document.addEventListener('click', function(event) {
    const profileIcon = document.getElementById('profileIcon');
    const dropdown = document.getElementById('profileDropdown');
    if (!profileIcon.contains(event.target) && !dropdown.contains(event.target)) {
        dropdown.classList.remove('show');
    }
});

// Edit Username (placeholder)
function editUsername() {
    const newName = prompt('Enter new username:');
    if (newName) {
        // You can add AJAX/Flask call here to update username in backend
        alert('Username updated to: ' + newName);
        // Optionally update avatar
        document.querySelector('.profile-icon img').src = `https://ui-avatars.com/api/?name=${encodeURIComponent(newName)}&background=667eea&color=fff&rounded=true`;
    }
}
// --- Logout: Show login page and hide nav/dashboard ---
function logout() {
    // If loginPage exists, show it and hide nav/dashboard
    const loginPage = document.getElementById('loginPage');
    if (loginPage) {
        // Hide all pages
        document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
        loginPage.classList.add('active');
        document.getElementById('navBar').classList.remove('show');
    } else {
        // If no login page, reload or redirect to login route
        window.location.href = '/login';
    }
}
// --- User Budget Input Logic ---
document.addEventListener('DOMContentLoaded', function() {
    // ...existing code...
    // User Budget Form logic
    const userBudgetForm = document.getElementById('userBudgetForm');
    if (userBudgetForm) {
        userBudgetForm.addEventListener('submit', function(e) {
            e.preventDefault();
            // Get values
            const month = document.getElementById('inputMonth').value || '--';
            const budget = document.getElementById('inputBudget').value || '--';
            const rent = parseInt(document.getElementById('inputRent').value) || 0;
            const grocery = parseInt(document.getElementById('inputGrocery').value) || 0;
            const travel = parseInt(document.getElementById('inputTravel').value) || 0;
            // Output
            document.getElementById('outputMonth').textContent = month;
            document.getElementById('outputBudget').textContent = budget !== '--' ? `₹${parseInt(budget).toLocaleString()}` : '--';
            document.getElementById('outputInitialExpense').textContent = `₹${(rent + grocery + travel).toLocaleString()}`;
        });
    }
});
// BudgetWise Pro - Dashboard Only (No Login)
// All login logic removed. Dashboard is default. Navigation is open.

// Expense data
let expenseData = {
    'Food': 15000,
    'Transport': 8000,
    'Education': 12000,
    'Loans': 20000,
    'Rent': 25000,
    'Entertainment': 5000,
    'Healthcare': 7000,
    'Shopping': 6000
};

const monthlyBudget = 120000;
let barChart, pieChart;

function createParticles() {
    const particlesContainer = document.getElementById('particles');
    const particleCount = 50;
    for (let i = 0; i < particleCount; i++) {
        const particle = document.createElement('div');
        particle.className = 'particle';
        particle.style.left = Math.random() * 100 + '%';
        particle.style.top = Math.random() * 100 + '%';
        particle.style.animationDelay = Math.random() * 6 + 's';
        particle.style.animationDuration = (6 + Math.random() * 4) + 's';
        particlesContainer.appendChild(particle);
    }
}

function showPage(pageId) {
    // Hide all pages
    const pages = document.querySelectorAll('.page');
    pages.forEach(page => page.classList.remove('active'));
    // Show selected page
    document.getElementById(pageId + 'Page').classList.add('active');
    // Initialize specific page content
    if (pageId === 'reports') {
        setTimeout(() => {
            initCharts();
            updateReportsStats();
        }, 100);
    } else if (pageId === 'expenses') {
        populateExpenseTable();
    } else if (pageId === 'dashboard') {
        updateDashboardStats();
    }
}

function addExpense(event) {
    event.preventDefault();
    const category = document.getElementById('expenseCategory').value;
    const amount = parseInt(document.getElementById('expenseAmount').value);
    if (category && amount > 0) {
        if (expenseData[category]) {
            expenseData[category] += amount;
        } else {
            expenseData[category] = amount;
        }
        const successMsg = document.getElementById('addExpenseSuccess');
        successMsg.style.display = 'block';
        setTimeout(() => { successMsg.style.display = 'none'; }, 3000);
        document.getElementById('expenseCategory').value = '';
        document.getElementById('expenseAmount').value = '';
        populateExpenseTable();
        updateDashboardStats();
    }
}

function deleteExpenseCategory(category) {
    if (confirm(`Are you sure you want to delete all expenses for ${category}?`)) {
        delete expenseData[category];
        populateExpenseTable();
        updateDashboardStats();
    }
}

function updateDashboardStats() {
    const total = Object.values(expenseData).reduce((a, b) => a + b, 0);
    const categoryCount = Object.keys(expenseData).length;
    const dailyAverage = Math.round(total / 30);
    const remaining = monthlyBudget - total;
    const progressPercentage = Math.min((total / monthlyBudget) * 100, 100);
    document.getElementById('totalExpenses').textContent = `₹${total.toLocaleString()}`;
    document.getElementById('categoryCount').textContent = categoryCount;
    document.getElementById('dailyAverage').textContent = `₹${dailyAverage.toLocaleString()}`;
    document.getElementById('budgetSpent').textContent = total.toLocaleString();
    document.getElementById('budgetTotal').textContent = monthlyBudget.toLocaleString();
    document.getElementById('budgetRemaining').textContent = remaining.toLocaleString();
    document.getElementById('budgetProgress').style.width = `${progressPercentage}%`;
    // Progress bar color
    const progressBar = document.getElementById('budgetProgress');
    if (progressPercentage > 90) {
        progressBar.style.background = 'linear-gradient(135deg, #f44336 0%, #ff5722 100%)';
    } else if (progressPercentage > 75) {
        progressBar.style.background = 'linear-gradient(135deg, #ff9800 0%, #ffc107 100%)';
    } else {
        progressBar.style.background = 'linear-gradient(135deg, #4caf50 0%, #66bb6a 100%)';
    }
}

function updateReportsStats() {
    const total = Object.values(expenseData).reduce((a, b) => a + b, 0);
    const sortedExpenses = Object.entries(expenseData).sort(([,a], [,b]) => b - a);
    const highest = sortedExpenses[0];
    const lowest = sortedExpenses[sortedExpenses.length - 1];
    document.getElementById('reportsTotalExpenses').textContent = `₹${total.toLocaleString()}`;
    document.getElementById('reportsHighestCategory').textContent = `₹${highest[1].toLocaleString()}`;
    document.getElementById('reportsLowestCategory').textContent = `₹${lowest[1].toLocaleString()}`;
    updateInsights();
}

function updateInsights() {
    const total = Object.values(expenseData).reduce((a, b) => a + b, 0);
    const rentPercentage = ((expenseData['Rent'] || 0) / total * 100).toFixed(1);
    const essentialCategories = ['Rent', 'Food', 'Transport'];
    const essentialTotal = essentialCategories.reduce((sum, cat) => sum + (expenseData[cat] || 0), 0);
    const essentialPercentage = ((essentialTotal / total) * 100).toFixed(1);
    const savingAmount = monthlyBudget - total;
    const savingPercentage = ((savingAmount / monthlyBudget) * 100).toFixed(1);
    const insights = [
        `<strong>Rent</strong> takes up ${rentPercentage}% of your total expenses - ${rentPercentage < 30 ? 'within recommended range' : 'consider optimizing'}`,
        `Essential expenses (Rent + Food + Transport) account for ₹${essentialTotal.toLocaleString()} (${essentialPercentage}% of total)`,
        `Your entertainment spending is ${((expenseData['Entertainment'] || 0) / (expenseData['Food'] || 1) * 100).toFixed(0)}% of your food budget`,
        `You're saving ₹${savingAmount.toLocaleString()} monthly which is ${savingPercentage}% of budget - ${savingPercentage > 15 ? 'excellent saving rate!' : 'consider increasing savings'}`
    ];
    document.getElementById('insightsList').innerHTML = insights.map(insight => `<li>${insight}</li>`).join('');
}

function initCharts() {
    const categories = Object.keys(expenseData);
    const amounts = Object.values(expenseData);
    const colors = [
        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF',
        '#4ECDC4', '#45B7D1'
    ];
    if (barChart) barChart.destroy();
    if (pieChart) pieChart.destroy();
    const barCtx = document.getElementById('barChart').getContext('2d');
    barChart = new Chart(barCtx, {
        type: 'bar',
        data: {
            labels: categories,
            datasets: [{
                label: 'Amount (₹)',
                data: amounts,
                backgroundColor: colors.slice(0, categories.length),
                borderColor: colors.slice(0, categories.length),
                borderWidth: 2,
                borderRadius: 8,
                borderSkipped: false,
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const total = amounts.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed.y / total) * 100).toFixed(1);
                            return `₹${context.parsed.y.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    grid: { color: 'rgba(0, 0, 0, 0.1)' },
                    ticks: {
                        callback: function(value) {
                            return '₹' + value.toLocaleString();
                        }
                    }
                },
                x: { grid: { display: false } }
            },
            animation: { duration: 2000, easing: 'easeInOutQuart' }
        }
    });
    const pieCtx = document.getElementById('pieChart').getContext('2d');
    pieChart = new Chart(pieCtx, {
        type: 'doughnut',
        data: {
            labels: categories,
            datasets: [{
                data: amounts,
                backgroundColor: colors.slice(0, categories.length),
                borderColor: '#fff',
                borderWidth: 3,
                hoverBorderWidth: 5
            }]
        },
        options: {
            responsive: true,
            cutout: '60%',
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true,
                        font: { size: 12 }
                    }
                },
                tooltip: {
                    backgroundColor: 'rgba(0, 0, 0, 0.8)',
                    titleColor: 'white',
                    bodyColor: 'white',
                    borderColor: 'rgba(255, 255, 255, 0.2)',
                    borderWidth: 1,
                    callbacks: {
                        label: function(context) {
                            const total = amounts.reduce((a, b) => a + b, 0);
                            const percentage = ((context.parsed / total) * 100).toFixed(1);
                            return `${context.label}: ₹${context.parsed.toLocaleString()} (${percentage}%)`;
                        }
                    }
                }
            },
            animation: { animateRotate: true, duration: 2000 }
        }
    });
}

function populateExpenseTable() {
    const tbody = document.getElementById('expenseTableBody');
    const total = Object.values(expenseData).reduce((a, b) => a + b, 0);
    let html = '';
    const sortedExpenses = Object.entries(expenseData).sort(([,a], [,b]) => b - a);
    sortedExpenses.forEach(([category, amount]) => {
        const percentage = ((amount / total) * 100).toFixed(1);
        html += `
            <tr>
                <td><strong>${getCategoryIcon(category)} ${category}</strong></td>
                <td>₹${amount.toLocaleString()}</td>
                <td>${percentage}%</td>
                <td>
                    <button onclick="deleteExpenseCategory('${category}')" 
                            class="btn" style="padding: 8px 16px; font-size: 14px; background: linear-gradient(135deg, #ff6b6b 0%, #ff5252 100%);">
                        🗑️ Delete
                    </button>
                </td>
            </tr>
        `;
    });
    html += `
        <tr style="background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%); font-weight: bold; border-top: 3px solid #667eea;">
            <td><strong>TOTAL</strong></td>
            <td><strong>₹${total.toLocaleString()}</strong></td>
            <td><strong>100.0%</strong></td>
            <td>-</td>
        </tr>
    `;
    tbody.innerHTML = html;
}

function getCategoryIcon(category) {
    const icons = {
        'Food': '🍔',
        'Transport': '🚗',
        'Education': '📚',
        'Loans': '🏦',
        'Rent': '🏠',
        'Entertainment': '🎬',
        'Healthcare': '🏥',
        'Shopping': '🛒',
        'Other': '📝'
    };
    return icons[category] || '📝';
}

function downloadCSV() {
    const total = Object.values(expenseData).reduce((a, b) => a + b, 0);
    let csvContent = "data:text/csv;charset=utf-8,";
    csvContent += "Category,Amount (₹),Percentage\n";
    Object.entries(expenseData).forEach(([category, amount]) => {
        const percentage = ((amount / total) * 100).toFixed(1);
        csvContent += `${category},${amount},${percentage}%\n`;
    });
    csvContent += `Total,${total},100.0%\n`;
    csvContent += `Budget,${monthlyBudget},-\n`;
    csvContent += `Remaining,${monthlyBudget - total},-\n`;
    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", `budgetwise_report_${new Date().toISOString().split('T')[0]}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
}

function downloadPDF() {
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    const total = Object.values(expenseData).reduce((a, b) => a + b, 0);
    doc.setFontSize(24);
    doc.setFont(undefined, 'bold');
    doc.text('BudgetWise Pro - Expense Report', 20, 30);
    doc.setFontSize(12);
    doc.setFont(undefined, 'normal');
    doc.text(`Generated: ${new Date().toLocaleDateString()}`, 20, 45);
    doc.text(`Total Expenses: ₹${total.toLocaleString()}`, 20, 55);
    doc.text(`Monthly Budget: ₹${monthlyBudget.toLocaleString()}`, 20, 65);
    doc.text(`Remaining Budget: ₹${(monthlyBudget - total).toLocaleString()}`, 20, 75);
    doc.setFontSize(16);
    doc.setFont(undefined, 'bold');
    doc.text('Detailed Breakdown:', 20, 95);
    doc.setFontSize(12);
    doc.setFont(undefined, 'normal');
    let yPosition = 110;
    const sortedExpenses = Object.entries(expenseData).sort(([,a], [,b]) => b - a);
    sortedExpenses.forEach(([category, amount]) => {
        const percentage = ((amount / total) * 100).toFixed(1);
        doc.text(`${category}:`, 30, yPosition);
        doc.text(`₹${amount.toLocaleString()} (${percentage}%)`, 100, yPosition);
        yPosition += 12;
    });
    yPosition += 20;
    doc.setFont(undefined, 'bold');
    doc.text('Budget Analysis:', 20, yPosition);
    doc.setFont(undefined, 'normal');
    yPosition += 15;
    const budgetUsed = ((total / monthlyBudget) * 100).toFixed(1);
    doc.text(`Budget Utilization: ${budgetUsed}%`, 30, yPosition);
    yPosition += 12;
    if (budgetUsed > 90) {
        doc.text('Status: High spending - Review expenses', 30, yPosition);
    } else if (budgetUsed > 75) {
        doc.text('Status: Moderate spending - Monitor closely', 30, yPosition);
    } else {
        doc.text('Status: Good spending - Within limits', 30, yPosition);
    }
    doc.save(`budgetwise_report_${new Date().toISOString().split('T')[0]}.pdf`);
}

document.addEventListener('DOMContentLoaded', function() {
    createParticles();
    showPage('dashboard');
    document.getElementById('navBar').classList.add('show');
    populateExpenseTable();
    updateDashboardStats();
});

document.addEventListener('mousemove', function(e) {
    const particles = document.querySelectorAll('.particle');
    const mouseX = e.clientX / window.innerWidth;
    const mouseY = e.clientY / window.innerHeight;
    particles.forEach((particle, index) => {
        const speed = (index % 3 + 1) * 0.5;
        const x = mouseX * speed;
        const y = mouseY * speed;
        particle.style.transform = `translate(${x}px, ${y}px)`;
    });
});
