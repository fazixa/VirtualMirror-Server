let openCam = () => {
    // axios.post('/open-cam')
    //     .then(res => {
    //         console.log(res)
    //     })
    //     .catch(err => {
    //         console.log(err)
    //     })
    $( "#container" ).append(`<img id="vid-feed" src="/video-feed" width="100%" alt="video-feed"></img>`);
}

let closeCam = () => {
    // axios.post('/close-cam')
    //     .then(res => {
    //         console.log(res)
    //     })
    //     .catch(err => {
    //         console.log(err)
    //     })
    $( "#vid-feed" ).remove();
}

let blush = () => {
    axios.get('/blush')
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

let eyeshadow = () => {
    axios.get('/eyeshadow')
        .then(res => {
            console.log(res)
        })
        .catch(err => {
            console.log(err)
        })
}

// let toggle_feed =