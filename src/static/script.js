var slideIndex = [];
var slideId = [];
var last_step = 2;
var last_slide = 0;

function load_slides(num) {
    for (i=1; i<=num; i++) {
        slideIndex.push(1)
        slideId.push("mySlides"+i)
    }
    for (i=0; i<num; i++) {
        showSlides(1,i)
    }
}

function plusSlides(n, no) {
    showSlides(slideIndex[no] += n, no);
}

function showSlides(n, no) {
    var i;
    var x = document.getElementsByClassName(slideId[no]);
    if (n > x.length) {slideIndex[no] = 1}
    if (n < 1) {slideIndex[no] = x.length}
    for (i = 0; i < x.length; i++) {
         x[i].style.display = "none";
    }
    x[slideIndex[no]-1].style.display = "block";
    var id = x[slideIndex[no]-1].parentElement.id;
    // slideIndex[no]-1 + 2 steps
    document.getElementById("span_"+id).innerHTML = "&emsp;&emsp;"+id+" (step "+(slideIndex[no]+1)+")";
}

window.onload = function() {
    // setup the button click
    document.getElementById("run").onclick = function() {
        run_program();
    };
}

function get_started() {
    data_path();
    console.log(getCookie("path"));
    document.getElementById("image_frame").src = getCookie("path") + "/pages/all_images.html";
    document.getElementById("out").src = getCookie("path") + "/output.txt";
    setCookie("is_running", "False", 1)

    $.post('html_slideshow', "", function(data, status) {
        setCookie("num_images", data[0], 1)
        document.getElementById("step_controller1").innerHTML += data[1]
        document.getElementById("step_controller2").innerHTML += data[2]
        document.getElementById("slides").innerHTML += data[3] // also loads the selector (see functions in html_helper.py file for more details)
    });
    setTimeout(function() {load_slides(getCookie("num_images"));}, 1000); // ensures the innerHTML content has loaded before continuing
    setTimeout(function() {reload()}, 2000); // refreshes for image loading
    setTimeout(function() {displayOne()}, 1000); // refreshes for image loading
    setTimeout(function() {add_stepper()}, 1000); // refreshes for image loading
}
get_started()

function reload_output() {
    document.getElementById('out').src = document.getElementById('out').src;
}
// setInterval(reload_output, 1000)

function reload_images() {
    document.getElementById('image_frame').src = document.getElementById('image_frame').src;
}

function add_buttons() {
    text_helper_buttons(5);
    text_helper_switches(2);
    for (i=0; i<5; i++) {
        document.getElementById("input").innerHTML += getCookie("button_result"+i);
    }
    for (i=0; i<2; i++) {
        document.getElementById("input").innerHTML += getCookie("toggle_result"+i);
    }
    add_buttons_helper();
    add_switches_helper();
}
add_buttons();

function run_program() {
    if (getCookie("is_running") == 'False') {
        console.log("running")
        setCookie("is_running", "True", 1)
        args = ["run"];
        args = args.concat(get_button_values());
        args = args.concat(get_toggle_values());
        console.log(args.join());
        // ajax the JSON to the server
        $.post("receiver", args.join(), function(data, status) {
            // once program has finished running
            setCookie("is_running", "False", 1)
        });
        // stop link reloading the page
        event.preventDefault();
    } else {
        alert("Error! Program is still running!")
    }

}

function data_path() {
    // ajax the JSON to the server
    $.post("helper", "", function(data, status){
        setCookie("path", data + '/data', 1);  // cookie stored for one day
        console.log(getCookie("path"));
    });
    // stop link reloading the page
    if (!(event === undefined)) {
        event.preventDefault();
    }
}

function setCookie(cname, cvalue, exdays) {
    var d = new Date();
    d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
    var expires = "expires="+d.toUTCString();
    document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}

function add_buttons_helper() {
    var slider1 = document.getElementById("range1");
    var output1 = document.getElementById("demo1");
    output1.innerHTML = slider1.value;
    slider1.oninput = function() {
        output1.innerHTML = slider1.value;
    }
    var slider2 = document.getElementById("range2");
    var output2 = document.getElementById("demo2");
    output2.innerHTML = slider2.value;
    slider2.oninput = function() {
        output2.innerHTML = slider2.value;
    }
    var slider3 = document.getElementById("range3");
    var output3 = document.getElementById("demo3");
    output3.innerHTML = slider3.value;
    slider3.oninput = function() {
        output3.innerHTML = slider3.value;
    }
    var slider4 = document.getElementById("range4");
    var output4 = document.getElementById("demo4");
    output4.innerHTML = slider4.value;
    slider4.oninput = function() {
        output4.innerHTML = slider4.value;
    }
    var slider5 = document.getElementById("range5");
    var output5 = document.getElementById("demo5");
    output5.innerHTML = slider5.value;
    slider5.oninput = function() {
        output5.innerHTML = slider5.value;
    }
    var slider6 = document.getElementById("range6");
    var output6 = document.getElementById("demo6");
    output6.innerHTML = slider6.value;
    slider6.oninput = function() {
        output6.innerHTML = slider6.value;
    }
    var slider7 = document.getElementById("range7");
    var output7 = document.getElementById("demo7");
    output7.innerHTML = slider7.value;
    slider7.oninput = function() {
        output7.innerHTML = slider7.value;
    }
    var slider8 = document.getElementById("range8");
    var output8 = document.getElementById("demo8");
    output8.innerHTML = slider8.value;
    slider8.oninput = function() {
        output8.innerHTML = slider8.value;
    }
    var slider9 = document.getElementById("range9");
    var output9 = document.getElementById("demo9");
    output9.innerHTML = slider9.value;
    slider9.oninput = function() {
        output9.innerHTML = slider9.value;
    }
    var slider10 = document.getElementById("range10");
    var output10 = document.getElementById("demo10");
    output10.innerHTML = slider10.value;
    slider10.oninput = function() {
        output10.innerHTML = slider10.value;
    }
}

function add_switches_helper() {
	var switches1 = document.getElementById("toggle1");
	var s_val1 = document.getElementById("switch1");
	s_val1.innerHTML = switches1.checked ? "True" : "False";
	switches1.oninput = function() {
		s_val1.innerHTML = switches1.checked ? "True" : "False";
	}
	var switches2 = document.getElementById("toggle2");
	var s_val2 = document.getElementById("switch2");
	s_val2.innerHTML = switches2.checked ? "True" : "False";
	switches2.oninput = function() {
		s_val2.innerHTML = switches2.checked ? "True" : "False";
	}
	var switches3 = document.getElementById("toggle3");
	var s_val3 = document.getElementById("switch3");
	s_val3.innerHTML = switches3.checked ? "True" : "False";
	switches3.oninput = function() {
		s_val3.innerHTML = switches3.checked ? "True" : "False";
	}
	var switches4 = document.getElementById("toggle4");
	var s_val4 = document.getElementById("switch4");
	s_val4.innerHTML = switches4.checked ? "True" : "False";
	switches4.oninput = function() {
		s_val4.innerHTML = switches4.checked ? "True" : "False";
	}
}

function get_button_values() {
    var values = [];
    for (i=1; i<=10; i++) {
        values.push(document.getElementById("range"+i).value);
    }
    console.log(values);
    return values;
}

function get_toggle_values() {
    var values = [];
    for (i=1; i<=4; i++) {
        values.push(document.getElementById("toggle"+i).checked ? "True" : "False");
    }
    console.log(values);
    return values;
}


function text_helper_buttons(num) {
    // ajax the JSON to the server
    $.post("html_buttons", "", function(data, status){
        for (i=0; i<num; i++) {
            setCookie("button_result"+i, data['result'][i], 1);  // cookie stored for one day
        }
    });
    // stop link reloading the page
    if (!(event === undefined)) {
        event.preventDefault();
    }
}

function text_helper_switches(num) {
    // ajax the JSON to the server
    $.post("html_switch", "", function(data, status){
        console.log(data)
        //console.log('here '+data['result'])
        for (i=0; i<num; i++) {
            setCookie("toggle_result"+i, data['result'][i], 1);  // cookie stored for one day
            console.log('set cookie!')
            console.log(getCookie("toggle_result"+i))
        }
    });
    // stop link reloading the page
    if (!(event === undefined)) {
        event.preventDefault();
    }
}

function add_css() {
    var cssId = 'myCss';  // you could encode the css path itself to generate id..
    if (!document.getElementById(cssId))
    {
        var head  = document.getElementsByTagName('head')[0];
        var link  = document.createElement('link');
        link.id   = cssId;
        link.rel  = 'stylesheet';
        link.type = 'text/css';
        link.href = getCookie("path") + "/src/static/format.css";
        link.media = 'all';
        head.appendChild(link);
    }
}

function add_stepper() {
  var stepper = document.getElementById("stepper");
  var step_info = document.getElementById("step_info");
  step_info.innerHTML = stepper.value;
  last_step = 2;
  stepper.oninput = function() {
      step_info.innerHTML = stepper.value;
      plusSlides(stepper.value - last_step, last_slide)
      last_step = stepper.value;
  }
}

function displayOne() {
    var e = document.getElementById("chooser");
    var graph_name = e.options[e.selectedIndex].text;
    last_slide = e.options[e.selectedIndex].value;
    var x = document.getElementsByClassName("slideshow-container")

    for (i=0; i<getCookie("num_images"); i++) {
        if (x[i].id != graph_name) {
            x[i].style.display = "none"
        } else {
            x[i].style.display = "block"
        }
    }

    // reset to step 2 (changing value will trigger on input)
    document.getElementById("stepper").value = 2;
    document.getElementById("step_info").innerHTML = 2;
}

function reload() {
    if (!window.location.hash) {
        window.location = window.location + '#loaded';
        window.location.reload();
    }
}
