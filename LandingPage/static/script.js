// Typing animation
const textElement = document.getElementById("typing-text");
const text = "Welcome to Our AI Expense Forecasting Page";
let index = 0;
let isDeleting = false;

function typeEffect() {
  if (!isDeleting && index <= text.length) {
    textElement.textContent = text.slice(0, index++);
  } else if (isDeleting && index >= 0) {
    textElement.textContent = text.slice(0, index--);
  }

  if (index === text.length) {
    isDeleting = true;
    setTimeout(typeEffect, 1500);
  } else if (isDeleting && index === 0) {
    isDeleting = false;
    setTimeout(typeEffect, 500);
  } else {
    setTimeout(typeEffect, isDeleting ? 80 : 120);
  }
}

typeEffect();

// Line Chart
const ctx = document.getElementById('lineChart').getContext('2d');
new Chart(ctx, {
  type: 'line',
  data: {
    labels: ["Jan", "Feb", "Mar", "Apr", "May", "Jun"],
    datasets: [
      {
        label: "Income",
        data: [1200, 1400, 1500, 1700, 1600, 1800],
        borderColor: "#ff80ff",
        fill: false,
        tension: 0.4
      },
      {
        label: "Expenses",
        data: [800, 950, 1100, 1200, 1300, 1250],
        borderColor: "#80d0ff",
        fill: false,
        tension: 0.4
      }
    ]
  },
  options: {
    responsive: true,
    plugins: {
      legend: { labels: { color: "#fff" } }
    },
    scales: {
      x: { ticks: { color: "#fff" } },
      y: { ticks: { color: "#fff" } }
    }
  }
});
