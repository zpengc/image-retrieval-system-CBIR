//The on() method attaches one or more event handlers for the selected elements
$(window).on('load', function () {
    if ($(".image-target img").length) {
        let size = get_size($(".image-target img"));
        $(".image-info").append("：" + size.width + " × " + size.height + "</p>");
    }
    $(".image-grid .image-item").each(function () {
        let img = $(this).children("img");
        let div = $(this).children("div");
        if (img.length) {
            let size = get_size(img);
            div.append(size.width + " × " + size.height);
        }
    });
    $("body").show();
});




function get_size(img) {
    return {
        'width': img.get(0).naturalWidth,
        'height': img.get(0).naturalHeight
    };
}
