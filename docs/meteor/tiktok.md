<script type="text/javascript" src="meteor.js"></script>

<script type="text/javascript">
    window.onload = function() {
        var restiktok = {
            canvasID : "canvastiktok",
            audioID : "audiotiktok",
            dataPath : "tiktok.meteor"
        };
        meteortiktok = new Meteor(restiktok);
    };
</script>

<style type="text/css">
    canvas {
        display: block;
        width: 100%;
        height: width*0.5625;
    }		
</style>

<div>
    <canvas id="canvastiktok" width="800" height="450"></canvas>
</div>
<div>
    <audio id='audiotiktok' controls="controls">
        <source type="audio/mpeg" src="tiktok.mp3"/>
    </audio>
</div>

# Test


