from flask import Flask, render_template, request, redirect, url_for, flash

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages


@app.route("/")
def home():
    return render_template("index.html")


# Login page (GET) and form handler (POST)
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        # Dummy check (replace with real authentication)
        if username and password:
            # On successful login, redirect to home.html
            return redirect(url_for("user_home"))
        else:
            flash("Invalid username or password.")
    return render_template("login.html")

# Route for user home (after login)
@app.route("/user_home")
def user_home():
    return render_template("home.html")


# SignUp page (GET) and form handler (POST)
@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        gender = request.form.get("gender")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")
        if password != confirm_password:
            flash("Passwords do not match.")
            return render_template("signup.html")
        # Dummy registration logic (replace with DB logic)
        if username and email and gender and password:
            flash("Registration successful! Please log in.")
            return redirect(url_for("login"))
        else:
            flash("Please fill all fields.")
    return render_template("signup.html")

if __name__ == "__main__":
    app.run(debug=True)
