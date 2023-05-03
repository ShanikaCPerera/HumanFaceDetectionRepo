function fileValidation() {
		var fileInput =
			document.getElementById('imageFile');

		var filePath = fileInput.value;

		// Allowing file type
		var allowedExtensions =
				/(\.jpg|\.jpeg|\.png)$/i;
		if((fileInput.files[0].size/ 1024) > 500)
		{
			document.getElementById("load-button").disabled = true;
			alert('File too big: select an image of 500Kb or less');
			fileInput.value = '';
			return false;
		}
		if (!allowedExtensions.exec(filePath)) {
			document.getElementById("load-button").disabled = true;
			alert('Invalid file type');
			fileInput.value = '';
			return false;
		}
		else
		{


			document.getElementById("load-button").disabled = false;
			// Image preview
			if (fileInput.files && fileInput.files[0]) {
				var reader = new FileReader();
				reader.onload = function(e) {
					document.getElementById(
						'imagePreview').innerHTML =
						'<div class="card-body"><h1 class="card-title">Original Image</h1></div><img src="' + e.target.result
						+ '" style="float:right" class="card-img-top"   />';
				};

				reader.readAsDataURL(fileInput.files[0]);
			}

		}
	}



	 function ElaborateImage() {

	   var file = document.getElementById('imageFile');

	   if (file.files.length == 0)
	   {
		alert('Select an Image');
	   }
	   else
	   {
		document.getElementById("div_spinner").style.display = "block";

		window.location = "http://127.0.0.1:2000/home";

	   }
	}




