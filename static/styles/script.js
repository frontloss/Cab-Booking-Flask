function populateSelect(min,max,id){
var select = document.getElementById(id);

for (var i = min; i<=max; i++)
{
    var opt = document.createElement('option');
    opt.value = i;
    opt.innerHTML = i;
    select.appendChild(opt);
}
}

function populateMonth(id){
var select = document.getElementById(id);
var arr = ['January','February','March','April','May','June','July','August','September','October','November','December']
for (var i = 0; i<arr.length; i++)
{
    var opt = document.createElement('option');
    opt.value = arr[i];
    opt.innerHTML = arr[i];
    select.appendChild(opt);
}
}

function populateWeekDay(id){
var select = document.getElementById(id);
var arr = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
for (var i = 0; i<arr.length; i++)
{
    var opt = document.createElement('option');
    opt.value = arr[i];
    opt.innerHTML = arr[i];
    select.appendChild(opt);
}	
}

