<!doctype html>
<html>
  <head>
    <script type="text/javascript" src="https://www.google.com/jsapi"></script>
    <script type="text/javascript">
      google.load("visualization", "1", {packages:["corechart"]});
      google.setOnLoadCallback(drawChart);
      function drawChart() {
	var data = google.visualization.arrayToDataTable([['Topic', 'Weight', 'This Topic Weight'],
	    {{HIST_ARRAY}}
	]);
	var options = {
	  cht: 'bvs',
	  title: 'Topic Weights',
	  legend: {position: 'none'},
	  chartArea: {left:40,top:40,width:"100%",height:"80%"},
	  vAxis: {title: 'Topic',  titleTextStyle: {color: 'red'}},
	  hAxis: {title: 'Occurrences',  titleTextStyle: {color: 'red'}}
	};

	var chart = new google.visualization.BarChart(document.getElementById('chart_div'));
	chart.draw(data, options);
	
	var handler = function(e) {
	  var sel = chart.getSelection();
	  var row = sel[0]['row'];
	  var url = "topics/topic_" + row + ".html";
	  window.location = url
	}
      google.visualization.events.addListener(chart, 'select', handler);
     }
    </script>
    
    <meta charset="utf-8">
    <meta http-equiv="X-UA-Compatible" content="chrome=1">
    <title>ROST Topic Browser</title>

    <link rel="stylesheet" href="./resource/stylesheets/styles.css">
    <link rel="stylesheet" href="./resource/stylesheets/pygment_trac.css">
    <meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
  </head>
  <body>
    <div class="wrapper">
      <header>
	<h1>ROST Visual Topic Model Browser</h1>
	<div id="chart_div" style="width: 325px; height: 500px;"></div>
      </header>
      <section>
	<p><h3>All Topics Single Maximum Likelihood Image</h3></p>
	{{#TOPIMAGE}}
	  <a href="topics/topic_{{TOPIC}}.html">Topic {{TOPIC}}</a>
	  <p><a href="./all-images.html#{{SEQ}}"><img src="./data/best/top{{TOPIC}}-seq{{SEQ}}.png"title="Seq. {{SEQ}}"></a></p>
	  
	{{/TOPIMAGE}}
	
      </section>
    </div>
  </body>
</html>
