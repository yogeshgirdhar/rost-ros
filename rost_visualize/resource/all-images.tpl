<!doctype html>
<html>
  <head>	
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="chrome=1">
    <title>ROST Topic Browser</title>

    <link rel="stylesheet" href="./resource/stylesheets/styles.css">
    <link rel="stylesheet" href="./resource/stylesheets/pygment_trac.css">
    <link rel="stylesheet" type="text/css" href="./resource/slider/slider.css" />
    
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
    <style type="text/css">
      .slider {
	width: 640px;
	font-family: Helvetica, Arial, sans-serif;
	font-size: 12px;
      }
    </style>

    <script type="text/javascript" src="./resource/slider/slider.js"></script>
    <script type="text/javascript">
    var im = new Image;
    var hist_data = new Array();
    hist_data = {{HIST_ARRAY}}
    window.onload = function(){
      var location = (parseInt(window.location.hash.split('#')[1]) - {{MIN_SEQ}})/{{SEQ_RANGE}};
      if (isNaN(location) || location < 0) location = 0;
      if (location > 1) location = 1;
      slider = new Slider('my-slider',
	{
	  steps: {{SEQ_RANGE}},
	  value: location,
	  snapping: true,
	  callback: function(value) {
	    var seq = Math.round({{MIN_SEQ}} + value * {{SEQ_RANGE}});
	    var index = seq - {{MIN_SEQ}};
	    var imgurl = "./data/all/" + seq + ".png";
	    var inhtml = "";
	    if (hist_data[index] && hist_data[index].length > 0){
	      for (var i=0; i<hist_data[index].length;i++){
		inhtml += "<a href=\./topics/topic_" + hist_data[index][i][0] + ".html\> Topic " + hist_data[index][i][0];
		inhtml += " (" + hist_data[index][i][1] +"%) </a>";
	      }
	    }
	    else{ inhtml = "No histogram data"; }
	    im.src = imgurl;
	    document.frame.src = imgurl;
	    document.frame.title = seq;
	    document.getElementById("seq_no").innerHTML = "Seq. No: " + seq + " - " + inhtml;
	  }
	});
      slider.callback(location)
    }
    </script>
  </head>
  <body>
    <div class="wrapper">
      <header><h1>ROST Visual Topic Model Browser</h1></header>
      <br> <br> <br>
      <p><h3>All Images</h3></p>
      <div id="seq_no">Seq. No: {{MIN_SEQ}}</div>
      <p><img src="./data/all/{{MIN_SEQ}}.png", name="frame" ></p>
      <div id="my-slider" class="slider">
      <div class="handle"> << - >> </div>
    </div>
  </body>
</html>