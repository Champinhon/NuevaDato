/* Theme Name: Dorsin - Responsive Landing page template
  Author: Themesbrand
  Version: 2.3.0
  File Description: Main JS file of the template
*/

//  Window scroll sticky class add

function windowScroll() {
    const navbar = document.getElementById("navbar");
    if (
        document.body.scrollTop >= 50 ||
        document.documentElement.scrollTop >= 50
    ) {
        navbar.classList.add("is-sticky");
    } else {
        navbar.classList.remove("is-sticky");
    }
}

window.addEventListener('scroll', (ev) => {
    ev.preventDefault();
    windowScroll();
})

// Navbar Active Class

var spy = new Gumshoe('#navbar-navlist a', {
    // Active classes
    navClass: 'active', // applied to the nav list item
    contentClass: 'active', // applied to the content
    offset: 70
});


// Smooth scroll 
var scroll = new SmoothScroll('#navbar-navlist a', {
    speed: 500,
    offset: 60
});


// Swicher
function toggleSwitcher() {
    var styleSwitcher = document.getElementById('style-switcher');
    if (styleSwitcher.style.left === "-168px") {
        styleSwitcher.style.left = "0";
    } else {
        styleSwitcher.style.left = "-168px";
    }
};

function setColor(theme) {
    document.getElementById('color-opt').href = '/static/css/colors/' + theme + '.css';
    toggleSwitcher(false);
};



// Contact Form
function validateForm() {
    var name = document.forms["myForm"]["name"].value;
    var email = document.forms["myForm"]["email"].value;
    var subject = document.forms["myForm"]["subject"].value;
    var comments = document.forms["myForm"]["comments"].value;
    document.getElementById("error-msg").style.opacity = 0;
    document.getElementById('error-msg').innerHTML = "";
    if (name == "" || name == null) {
        document.getElementById('error-msg').innerHTML = "<div class='alert alert-warning error_message'>Please enter a Name</div>";
        fadeIn();
        return false;
    }
    if (email == "" || email == null) {
        document.getElementById('error-msg').innerHTML = "<div class='alert alert-warning error_message'>Please enter a Email</div>";
        fadeIn();
        return false;
    }
    if (subject == "" || subject == null) {
        document.getElementById('error-msg').innerHTML = "<div class='alert alert-warning error_message'>Please enter a Subject</div>";
        fadeIn();
        return false;
    }
    if (comments == "" || comments == null) {
        document.getElementById('error-msg').innerHTML = "<div class='alert alert-warning error_message'>Please enter a Comments</div>";
        fadeIn();
        return false;
    }

    var xhttp = new XMLHttpRequest();
    xhttp.onreadystatechange = function () {
        if (this.readyState == 4 && this.status == 200) {
            document.getElementById("simple-msg").innerHTML = this.responseText;
            document.forms["myForm"]["name"].value = "";
            document.forms["myForm"]["email"].value = "";
            document.forms["myForm"]["subject"].value = "";
            document.forms["myForm"]["comments"].value = "";
        }
    };
    xhttp.open("POST", "php/contact.php", true);
    xhttp.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
    xhttp.send("name=" + name + "&email=" + email + "&subject=" + subject + "&comments=" + comments);
    return false;
}

function fadeIn() {
    var fade = document.getElementById("error-msg");
    var opacity = 0;
    var intervalID = setInterval(function () {
        if (opacity < 1) {
            opacity = opacity + 0.5
            fade.style.opacity = opacity;
        } else {
            clearInterval(intervalID);
        }
    }, 200);
}

// Tetx-rotate
var i = 0;
var text_rotate = [];
var time = 3000;

text_rotate[0] = "Professional Landing Page Template";
text_rotate[1] = "We help startups launch their products";
text_rotate[2] = "Perfect solution for small businesses";

function changeText() {
    document.getElementById("target").innerHTML = text_rotate[i];
    i = (i < text_rotate.length - 1) ? i + 1 : 0;
    setTimeout('changeText()', time);
}


//
/********************* light-dark js ************************/
//
const btn = document.getElementById("mode");
btn.addEventListener("click", (e) => {
    let theme = localStorage.getItem("theme");
    if (theme == "light" || theme == "") {
        document.body.setAttribute("data-bs-theme", "dark");
        localStorage.setItem("theme", "dark");
    } else {
        document.body.removeAttribute("data-bs-theme");
        localStorage.setItem("theme", "light");
    }
});

window.addEventListener('load', function () {
    toggleSwitcher();
});