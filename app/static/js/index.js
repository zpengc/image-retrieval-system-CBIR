$(document).ready(function () {
    // 提交输入的查询图像
    $("#fileimg").change(function (e) {
        let file = $(this)[0].files[0];
        if (validate_img(file))
            submit($("#imgform"));
    });

    // 提交输入的图像url
    $("#btnsubmit").click(function () {
        let url = $("#txturl").val();
        if (validate_url(url))
            submit($("#urlform"));
    });

    // 回车键提交输入的图像url
    $(document).keypress(function (e) {
        let key = e.which;
        // ASCII code 13 is carriage return
        if (key == 13) {
            e.preventDefault();
            $("#btnsubmit").click();
        }
    });

// parameter e comes from the javascript engine executing your callback function
    //Attach multiple event handlers using the map parameter
    // 拖拽检测区域是指定的小块区域
    $(".container").on({
        dragenter: function (e) {
//        prevents propagation of the same event from being called.
            e.stopPropagation();
//            cancels the event if it is cancelable, meaning that the default action that
//belongs to the event will not occur.
            e.preventDefault();
        },
        dragover: function (e) {
            e.stopPropagation();
            e.preventDefault();
        },
        drop: function (e) {
            e.stopPropagation();
            e.preventDefault();
            $("#dropzone").hide();
            $("#fileimg")[0].files = e.originalEvent.dataTransfer.files;
            if (validate_img($("#fileimg")[0].files[0]))
                submit($("#imgform"));
        }
    });

    // 拖拽检测区域是整个浏览器界面
    $(document).on({
        dragenter: function (e) {
            e.stopPropagation();
            e.preventDefault();
            // 拖拽文件刚进入浏览器界面就显示指定区域
            $("#dropzone").show();
        },
        dragover: function (e) {
            e.stopPropagation();
            e.preventDefault();
        },
        dragleave: function (e) {
            e.stopPropagation();
            e.preventDefault();
            if (e.clientX <= 0 ||
                e.clientX >= $(window).width() ||
                e.clientY <= 0 ||
                e.clientY >= $(window).height())
                $("#dropzone").hide();
        },
        drop: function (e) {
            e.stopPropagation();
            e.preventDefault();
            // 完全脱离浏览器界面再隐藏
            $("#dropzone").hide();
        }
    });

//清空url输入框，#表示id选择器
    $("#btnclose").click(function () {
        $("#txturl").val("");
    });
});

// 验证选择的查询图像
// PNG 图像是 "image/png"，TXT文件为 "text/plain"
function validate_img(file) {
    let type = file['type'];
    if (type.split('/')[0] != 'image') {
        alert("只接受图片格式的文件");
        return false;
    }
    else if (file.size >= 3 * 1024 * 1024) {
        alert("请上传小于3M的图片");
        return false;
    }
    return true;
}

// 验证输入的图像url
function validate_url(url) {
    let imgregex = /(https?:\/\/.*\.(?:png|jpg))/i;

    if (!url) {
        alert("图像URL为空!!!")
        return false;
    }
    else if (url.length > 1000) {
        alert("URL长度不超过1000");
        return false;
    }
    else if (!imgregex.test(url)) {
        alert("图片URL不合法");
        return false;
    }
    return true;
}

// 提交表单，图像表单和URL表单
function submit(form) {
    $("#uploadtip").show();
    try {
        form.submit();
    }
    catch (err) {
        alert(err);
        $("#uploadtip").hide();
    }
}