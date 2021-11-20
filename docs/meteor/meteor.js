
var blackPos=[1, 2, 4, 5, 6];
var whitePitchs=[0, 2, 4, 5, 7, 9, 11];
var blackPitchs=[1, 3, 6, 8, 10];
var keyPos = [0.5, 1.0, 1.5, 2.0, 2.5, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5];
var s_ColorBank = ["#418CF0","#FCB441","#DF3A02","#056492","#BFBFBF","#1A3B69","#FFE382","#129CDD","#CA6B4B","#005CDB","#F3D288","#506381","#F1B9A8","#E0830A","#7893BE"];

InStream = function(buffer)
{
	 this.offset = 0;
	 this.view = new DataView(buffer);
};

InStream.prototype.GetInt32=function()
{
	var data = this.view.getInt32(this.offset,true);
	this.offset+=4;
	return data;
}

InStream.prototype.GetUint32=function()
{
	var data = this.view.getUint32(this.offset,true);
	this.offset+=4;
	return data;
}

InStream.prototype.GetFloat32=function()
{
	var data = this.view.getFloat32(this.offset,true);
	this.offset+=4;
	return data;
}

InStream.prototype.GetUint8=function()
{
	var data = this.view.getUint8(this.offset,true);
	this.offset+=1;
	return data;
}

function Utf8ArrayToStr(array) {
    var out, i, len, c;
    var char2, char3;

    out = "";
    len = array.length;
    i = 0;
    while(i < len) {
    c = array[i++];
    switch(c >> 4)
    { 
      case 0: case 1: case 2: case 3: case 4: case 5: case 6: case 7:
        // 0xxxxxxx
        out += String.fromCharCode(c);
        break;
      case 12: case 13:
        // 110x xxxx   10xx xxxx
        char2 = array[i++];
        out += String.fromCharCode(((c & 0x1F) << 6) | (char2 & 0x3F));
        break;
      case 14:
        // 1110 xxxx  10xx xxxx  10xx xxxx
        char2 = array[i++];
        char3 = array[i++];
        out += String.fromCharCode(((c & 0x0F) << 12) |
                       ((char2 & 0x3F) << 6) |
                       ((char3 & 0x3F) << 0));
        break;
    }
    }

    return out;
}

InStream.prototype.GetString=function(len)
{
	var utf8Arr=new Array();
	for (var pos=0; pos<len; pos++)
	{
		var v=this.view.getUint8(this.offset+pos);
		utf8Arr.push(v);
	}
	this.offset+=len;
	return Utf8ArrayToStr(utf8Arr);
}

Meteor = function(resources)
{
	var scale= 1.0;
	if (resources.scale!=undefined)
		scale = resources.scale;

	this.cornerSize= 3.0*scale;
	this.whiteKeyWidth = 18.0*scale;
	this.blackKeyWidth = 14.0*scale;
	this.whiteKeyHeight = 80.0*scale;
	this.blackKeyHeight = 50.0*scale;
	this.whiteKeyPressedDelta = 3.0*scale;
	this.blackKeyPressedDelta = 2.0*scale;
	this.pressedLineWidth = 3.0;
	this.showTime = 1.0;
	this.meteorHalfWidth = 5.0*scale;
	this.percussion_flash_size_factor = 0.15;
	this.percussion_flash_limit = 0.3;
	this.singing_half_width = 8.0*scale;
	this.fontSize=30.0*scale;

	this.canvas=document.getElementById(resources.canvasID).getContext("2d");
	if (resources.audioID!=undefined)
		this.audio= document.getElementById(resources.audioID);

	this.notes=new Array();
	this.notes_sublists =	{
		minStart : 0.0,
		maxEnd : 0.0,
		interval : 0.0,
		subLists : new Array()
	};

	this.beats=new Array();
	this.beats_sublists = {
		minStart : 0.0,
		maxEnd : 0.0,
		interval : 0.0,
		subLists : new Array()
	};

	this.singings=new Array();
	this.singing_sublists = {
		minStart : 0.0,
		maxEnd : 0.0,
		interval : 0.0,
		subLists : new Array()
	};

	this.InstColorMap=new Array();
	this.PercColorMap=new Array();
	this.SingerColorMap=new Array();

	var that=this;

	var xhr = new XMLHttpRequest(); 
	xhr.open("GET", resources.dataPath); 
	xhr.responseType = "blob";
	xhr.onload = function() 
	{
		var myReader = new FileReader();
		myReader.readAsArrayBuffer(xhr.response);
		myReader.addEventListener("loadend", function(e)
	    {
	        var buffer = e.srcElement.result;
	        var inStream=new InStream(buffer);
	     
	        var count_notes= inStream.GetUint32();
	      
	        for (var i=0;i<count_notes; i++)
	    	{
	    		var note={
	    			instrumentId : inStream.GetUint32(),
	    			pitch : inStream.GetInt32(),
	    			start : inStream.GetFloat32(),
	    			end : inStream.GetFloat32()
	    		};
	    		that.notes.push(note);
	    	}

		    that.notes_sublists.minStart=inStream.GetFloat32();
        	that.notes_sublists.maxEnd=inStream.GetFloat32();
        	that.notes_sublists.interval=inStream.GetFloat32();
        	count_notes_sublists=inStream.GetUint32();

	    	for (var i=0;i<count_notes_sublists; i++)
		    {
		    	var count_indices=inStream.GetUint32();

		    	// console.log(count_indices);
		    	var sublist=new Array();
		    	for (var j=0; j<count_indices;j++)
	    			sublist.push(inStream.GetUint32());
		    	that.notes_sublists.subLists.push(sublist);
		    }	

		    var count_beats = inStream.GetUint32();

		    for (var i=0;i<count_beats; i++)
	    	{
	    		var beat={
	    			percId : inStream.GetUint32(),
	    			start : inStream.GetFloat32(),
	    			end : inStream.GetFloat32(),
	    			centerX : Math.random(),
	    			centerY : Math.random()
	    		};
	    		that.beats.push(beat);
	    	}

			that.beats_sublists.minStart=inStream.GetFloat32();
        	that.beats_sublists.maxEnd=inStream.GetFloat32();
        	that.beats_sublists.interval=inStream.GetFloat32();
        	var count_beats_sublists=inStream.GetUint32();

        	for (var i=0;i<count_beats_sublists; i++)
		    {
		    	var count_indices=inStream.GetUint32();

		    	// console.log(count_indices);
		    	var sublist=new Array();
		    	for (var j=0; j<count_indices;j++)
	    			sublist.push(inStream.GetUint32());
		    	that.beats_sublists.subLists.push(sublist);
		    }

		    var count_singings = inStream.GetUint32();

	    	for (var i=0;i<count_singings;i++)
	    	{
	    		var singerId = inStream.GetUint32();
	    		var len= inStream.GetUint8();
	    		var lyric=inStream.GetString(len);
	    		var pitchCount=inStream.GetUint32();
	    		var pitchData=new Array();
	    		for (var j=0;j<pitchCount;j++)
	    			pitchData.push(inStream.GetFloat32());
	    		var start=inStream.GetFloat32();
	    		var end=inStream.GetFloat32();
	    		var singing={
	    			singerId: singerId,
	    			lyric: lyric,
	    			pitch: pitchData,
	    			start : start,
	    			end :end
	    		};
	    		that.singings.push(singing);	    		
	    	}

	    	that.singing_sublists.minStart=inStream.GetFloat32();
        	that.singing_sublists.maxEnd=inStream.GetFloat32();
        	that.singing_sublists.interval=inStream.GetFloat32();
        	var count_singing_sublists=inStream.GetUint32();

        	for (var i=0;i<count_singing_sublists; i++)
		    {
		    	var count_indices=inStream.GetUint32();

		    	// console.log(count_indices);
		    	var sublist=new Array();
		    	for (var j=0; j<count_indices;j++)
	    			sublist.push(inStream.GetUint32());
		    	that.singing_sublists.subLists.push(sublist);
		    }

		    that.buildColorMap();
	    });
	}

	xhr.send();

	if (resources.capturer!=undefined)
	{
		this.capturer=resources.capturer;
		this.capturer.start();
	}

	this.startTime=performance.now();

	this.animationLoop();
};

Meteor.prototype.buildColorMap = function()
{
	var bankRef = 0;
	for (var beat of this.beats)
	{
		if (this.PercColorMap[beat.percId]==undefined)
		{
			this.PercColorMap[beat.percId]=s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= s_ColorBank.length) bankRef = 0;
		}
	}

	for (var singing of this.singings)
	{
		if (this.SingerColorMap[singing.singerId]==undefined)
		{
			this.SingerColorMap[singing.singerId]=s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= s_ColorBank.length) bankRef = 0;
		}
	}

	for (var note of this.notes)
	{
		if (this.InstColorMap[note.instrumentId]==undefined)
		{
			this.InstColorMap[note.instrumentId]=s_ColorBank[bankRef];
			bankRef++;
			if (bankRef >= s_ColorBank.length) bankRef = 0;
		}

	}
};

Meteor.prototype.draw_key = function(left, right, top, bottom, lineWidth, isBlack=false)
{
	if (isBlack)
	{
		this.canvas.fillStyle = '#000000';
		this.canvas.strokeStyle ='#FFFFFF';
	}
	else
	{
		this.canvas.fillStyle = '#FFFFFF';
		this.canvas.strokeStyle ='#000000';
	}
	this.canvas.lineWidth=lineWidth;
	this.canvas.beginPath();
	this.canvas.moveTo(left + this.cornerSize, top);
	this.canvas.lineTo(right - this.cornerSize, top);
	this.canvas.lineTo(right, top + this.cornerSize);
	this.canvas.lineTo(right, bottom - this.cornerSize);
	this.canvas.lineTo(right -this.cornerSize, bottom);
	this.canvas.lineTo(left + this.cornerSize, bottom);
	this.canvas.lineTo(left, bottom - this.cornerSize);
	this.canvas.lineTo(left, top + this.cornerSize);
	this.canvas.lineTo(left + this.cornerSize, top);
	this.canvas.closePath();
	this.canvas.fill();
	this.canvas.stroke();
};

Meteor.prototype.draw_flash = function(centerx, centery, radius, color, alpha)
{
	var gradient = this.canvas.createRadialGradient(centerx, centery, 0.0, centerx, centery, radius);
	gradient.addColorStop(0,color);
	gradient.addColorStop(1,"#000000");
	this.canvas.arc(centerx, centery, radius, 0, 2 * Math.PI);
	this.canvas.fillStyle = gradient;
	this.canvas.globalAlpha = alpha;
	this.canvas.fill();
	this.canvas.globalAlpha = 1.0;
};

function GetIntervalId(sublists, v)
{
	if (v<sublists.minStart) return 0;
	var id = Math.floor((v-sublists.minStart)/sublists.interval);
	if (id>=sublists.subLists.length)
		id = sublists.subLists.length -1;
	return id;
};

Meteor.prototype.draw = function(currentTime)
{
	this.canvas.fillStyle = "#000000";
	this.canvas.fillRect(0, 0, this.canvas.canvas.width, this.canvas.canvas.height);

	var note_inTime = currentTime;
	var note_intervalId = GetIntervalId(this.notes_sublists, note_inTime);
	var note_outTime = note_inTime - this.showTime;
	var note_intervalId_min = GetIntervalId(this.notes_sublists, note_outTime);

	/// draw meteors
	this.canvas.globalCompositeOperation="lighter";

	// notes
	if (this.notes_sublists.subLists.length>0)
	{
		var visiableNotes=new Set();
		for (var i = note_intervalId_min; i <= note_intervalId; i++)
		{
			var sublist= this.notes_sublists.subLists[i];
			for (var j = 0; j < sublist.length; j++)
			{
				var note = this.notes[sublist[j]]; 
				if (note.start<note_inTime && note.end> note_outTime)
					visiableNotes.add(note);
			}
		}

		for (var note of visiableNotes)
		{
			var startY = this.canvas.canvas.height- this.whiteKeyHeight-
			 (note.start - note_inTime) / -this.showTime* (this.canvas.canvas.height - this.whiteKeyHeight);
			var endY = this.canvas.canvas.height- this.whiteKeyHeight-
			 (note.end - note_inTime) / -this.showTime* (this.canvas.canvas.height - this.whiteKeyHeight);

			var pitch = note.pitch;
			var octave = 0;
			while (pitch < 0)
			{
				pitch += 12;
				octave--;
			}
			while (pitch >= 12)
			{
				pitch -= 12;
				octave++;
			}

			var x = this.canvas.canvas.width*0.5 + (octave*7.0 + keyPos[pitch])*this.whiteKeyWidth;

			var color=this.InstColorMap[note.instrumentId];
			
			this.canvas.beginPath();
			this.canvas.moveTo(x, startY);
			this.canvas.lineTo(x + this.meteorHalfWidth, startY+this.meteorHalfWidth);
			this.canvas.lineTo(x, endY);
			this.canvas.lineTo(x - this.meteorHalfWidth, startY+this.meteorHalfWidth);		
			this.canvas.closePath();

			var gradient = this.canvas.createLinearGradient(x, startY, x, endY);
			gradient.addColorStop(0,color);
			gradient.addColorStop(1,"#000000");
			this.canvas.fillStyle = gradient;
			this.canvas.fill();

		}

	}

	// beats
	if (this.beats_sublists.subLists.length>0)
	{
		var beat_intervalId = GetIntervalId(this.beats_sublists, note_inTime);
		var sublist= this.beats_sublists.subLists[beat_intervalId];
		for (var i = 0; i < sublist.length; i++)
		{
			var beat = this.beats[sublist[i]];
			var start = beat.start;
			var end = beat.end;

			// limting percussion flash time
			if (end - start > this.percussion_flash_limit)
				end = start + this.percussion_flash_limit;

			if (note_inTime >= start && note_inTime <= end)
			{
				var centerx = beat.centerX*this.canvas.canvas.width;
				var centery = beat.centerY*(this.canvas.canvas.height - this.whiteKeyHeight);
				var radius = this.canvas.canvas.width*this.percussion_flash_size_factor;

				var color=this.PercColorMap[beat.percId];
				var alpha = (end - note_inTime) / (end - start);

				this.draw_flash(centerx, centery, radius, color, alpha);
			}


		}
	}

	// singing
	if (this.singing_sublists.subLists.length>0)
	{
		var singing_intervalId = GetIntervalId(this.singing_sublists, note_inTime);
		var singing_intervalId_min = GetIntervalId(this.singing_sublists, note_outTime);

		var visiableNotes=new Set();
		for (var i = singing_intervalId_min; i <= singing_intervalId; i++)
		{
			var sublist= this.singing_sublists.subLists[i];
			for (var j = 0; j < sublist.length; j++)
			{
				var note = this.singings[sublist[j]]; 
				if (note.start<note_inTime && note.end> note_outTime)
					visiableNotes.add(note);
			}
		}

		var pixelPerPitch = this.whiteKeyWidth*7.0 / 12.0;

		for (var note of visiableNotes)
		{
			var startY = this.canvas.canvas.height- this.whiteKeyHeight-
			 (note.start - note_inTime) / -this.showTime* (this.canvas.canvas.height - this.whiteKeyHeight);
			var endY = this.canvas.canvas.height- this.whiteKeyHeight-
			 (note.end - note_inTime) / -this.showTime* (this.canvas.canvas.height - this.whiteKeyHeight);

			var color=this.SingerColorMap[note.singerId];
			var num_pitches = note.pitch.length;


			this.canvas.beginPath();
			if (0<num_pitches)
			{
				var x=note.pitch[0]*pixelPerPitch+this.canvas.canvas.width*0.5+this.whiteKeyWidth*0.5;
				var y=startY;
				this.canvas.moveTo(x-this.singing_half_width, y);
			}
			for (var i=1;i<num_pitches;i++)
			{
				var x=note.pitch[i]*pixelPerPitch+this.canvas.canvas.width*0.5+this.whiteKeyWidth*0.5;
				var k=i/(num_pitches - 1);
				var y=startY*(1.0 - k) + endY*k;
				this.canvas.lineTo(x-this.singing_half_width, y);
			}

			for (var i=num_pitches-1; i>=0; i--)
			{
				var x=note.pitch[i]*pixelPerPitch+this.canvas.canvas.width*0.5+this.whiteKeyWidth*0.5;
				var k=i/(num_pitches - 1);
				var y=startY*(1.0 - k) + endY*k;
				this.canvas.lineTo(x+this.singing_half_width, y);
			}

			this.canvas.closePath();
			var gradient = this.canvas.createLinearGradient(x, startY, x, endY);
			gradient.addColorStop(0,color);
			gradient.addColorStop(1,"#000000");
			this.canvas.fillStyle = gradient;
			this.canvas.fill();

			var x=note.pitch[0]*pixelPerPitch+this.canvas.canvas.width*0.5;
			this.canvas.fillStyle=color;
			this.canvas.font=this.fontSize.toFixed(0)+"px sans-serif";
			this.canvas.fillText(note.lyric,x+this.singing_half_width,startY);
		}


	}


	/// draw keyboard
	this.canvas.globalCompositeOperation="source-atop";

	var center=this.canvas.canvas.width*0.5;
	var octaveWidth = this.whiteKeyWidth*7.0;
	var minOctave = -Math.ceil(center/octaveWidth);
	var maxOctave = Math.floor(center/octaveWidth);	
	var numKeys = (maxOctave - minOctave + 1) * 12;
	var indexShift = -minOctave * 12;

	var pressed = new Array(numKeys);
	for (var i=0;i<numKeys;i++)
		pressed[i]=false;

	//notes
	if (this.notes_sublists.subLists.length>0)
	{
		var sublist= this.notes_sublists.subLists[note_intervalId];
		for (var i = 0; i < sublist.length; i++)
		{
			var note = this.notes[sublist[i]];
			var start = note.start;
			var end = note.end;

			end -= (end-start)*0.1;
			if (note_inTime >= start && note_inTime <= end)
			{
				var index = note.pitch + indexShift;
				if (index >= 0 && index < numKeys)
				{
					pressed[index] = true;
				}
			}

		}
	}

	for (var i = minOctave; center + i*octaveWidth < this.canvas.canvas.width; i++)
	{
		var octaveLeft = center + i*octaveWidth;
		for (var j = 0; j < 7; j++)
		{
			var index = whitePitchs[j] + i * 12 + indexShift;
			var keyPressed = pressed[index];

			var left = octaveLeft + j*this.whiteKeyWidth;
			var right = left + this.whiteKeyWidth;
			var bottom = keyPressed ? this.canvas.canvas.height- this.whiteKeyPressedDelta : this.canvas.canvas.height;
			var top = this.canvas.canvas.height- this.whiteKeyHeight;
			this.draw_key(left, right, top, bottom, keyPressed ? this.pressedLineWidth : 1.0);
		}
		for (var j = 0; j < 5; j++)
		{
			var index = blackPitchs[j] + i * 12 + indexShift;
			var keyPressed = pressed[index];

			var keyCenter = octaveLeft + blackPos[j] * this.whiteKeyWidth;
			var left = keyCenter - this.blackKeyWidth / 2.0;
			var right = keyCenter + this.blackKeyWidth / 2.0;

			var bottom = keyPressed ? this.canvas.canvas.height - this.whiteKeyHeight + this.blackKeyHeight - this.blackKeyPressedDelta : this.canvas.canvas.height - this.whiteKeyHeight + this.blackKeyHeight;
			var top = this.canvas.canvas.height - this.whiteKeyHeight;
			this.draw_key(	left, right, top, bottom, keyPressed ? this.pressedLineWidth : 1.0, true);
		}
	} 

}

Meteor.prototype.animationLoop = function() 
{
	var that = this;
    requestAnimationFrame(
    	function() 
    	{
    		that.animationLoop()
    	});

    if (this.audio!=undefined)
    	this.draw(this.audio.currentTime);
    else
    {
    	this.draw((performance.now()-this.startTime)*0.001);
    	if (this.capturer!=undefined)
    	{
    		this.capturer.capture(this.canvas.canvas);
    	}
    }
};
