const dropArea = document.querySelector('.dropArea'),
input = dropArea.querySelector('input');
const imgChosen = document.querySelector('#imgChosen');
const dropAreaChild = document.querySelector('.dropArea-child');
const timeSpan = document.querySelector('#time-span');
const imgBox = document.querySelector(".imgBox");
const cloud = document.querySelector(".cloud");
const btnSubmit = document.querySelector("#btn-submit");
const btnReload = document.querySelector("#btn-reload");
const buttonArea = document.querySelector(".buttonArea");
const imgResults = document.querySelectorAll(".img-result");
const imgResultsRow = document.querySelector(".img-result-row");
const imgNameResults = document.querySelectorAll(".img-result-name");
let file;
let jcropAPI;

dropArea.onclick = ()=>{
    input.click();
}

input.addEventListener("change", function (){
    file = this.files[0];
    if (!checkImageFile(file)){
        alert("Vui lòng tải lên file ảnh");
        return;
    }
    loadImage();
});

dropArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    dropAreaChild.classList.add("active");
});

dropArea.addEventListener("dragleave", () => {
    dropAreaChild.classList.remove("active");
});

dropArea.addEventListener("drop", (event) => {
    event.preventDefault();
    file = event.dataTransfer.files[0];
    if (!checkImageFile(file)){
        alert("Vui lòng tải lên file ảnh");
        return;
    }
    loadImage();
});

btnReload.onclick = (event) => {
    event.preventDefault();
    dropArea.classList.remove("d-none");
    imgBox.classList.add("d-none");
    buttonArea.classList.add("d-none");
    imgResultsRow.classList.add("d-none");
    timeSpan.classList.add("d-none");
}

btnSubmit.onclick = (event) => {
    event.preventDefault();
    upload();
}

function checkImageFile(file){
    let fileType = file.type;
    let validExt = ["image/jpeg", "image/jpg", "image/png"];
    return validExt.includes(fileType);
}

function loadImage() {
    if (jcropAPI)
        jcropAPI.destroy();
    dropArea.classList.add("d-none");
    imgResultsRow.classList.add("d-none");
    timeSpan.classList.add("d-none");
    let reader = new FileReader();
    reader.addEventListener("load", (event) => {
        imgChosen.src = event.target.result;

        $(function () {
            $('#imgChosen').Jcrop({
                onSelect: showCords
            }, function(){ jcropAPI = this });

            function showCords(c){
                cropImage(imgChosen, c.x, c.y, c.w, c.h);
            }
        })
    });
    reader.readAsDataURL(file);
    imgBox.classList.remove("d-none");
    buttonArea.classList.remove("d-none");
    dropAreaChild.classList.remove("active");
}

function cropImage(img, x, y, w, h) {
    let canvas = document.createElement("canvas");
    let scale = img.height / Number(window.getComputedStyle(img).getPropertyValue("height").replace('px',''));
    canvas.width = w * scale;
    canvas.height = h * scale;
    canvas.getContext("2d").drawImage(img, x * scale, y * scale, w * scale, h * scale, 0, 0, w * scale, h * scale);
    canvas.toBlob((blob) => { file = blob; });
}

function upload(){
    if (file){
        let fileForm = new FormData();
        fileForm.append('file', file);
        $.ajax({
            method: 'POST',
            url: '/',
            data: fileForm,
            processData: false,
            cache: false,
            contentType: false,
            success: function (response){
                if (response){
                    let results = response['results'];
                    for (let i = 0; i < Object.keys(results).length; i++){
                        imgResults[i].src = Object.keys(results)[i];
                        imgNameResults[i].innerHTML = Object.keys(results)[i].split("/").at(-1).split(".").at(0);
                    }
                    imgResultsRow.classList.remove("d-none");
                    timeSpan.innerHTML = "Time span: " + response['time'];
                    timeSpan.classList.remove("d-none");
                    timeSpan.scrollIntoView();
                }
                else{
                    console.log('No response');
                }
            },
            error: function(error){
                console.log(error);
            }
        });
    }
}

// Loading
$(document).ajaxStart(function(){
    $('#loading-screen').show();
});
$(document).ajaxComplete(function(){
    $('#loading-screen').hide();
});