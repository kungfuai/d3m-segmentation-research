<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
-->
[![LinkedIn][linkedin-shield]][linkedin-url]
[![MIT License][license-shield]][license-url]

<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://www.linkedin.com/company/kungfuai/">
    <img src="https://media-exp1.licdn.com/dms/image/C4E0BAQEgWgybqu6dDg/company-logo_200_200/0?e=1611187200&v=beta&t=svIQxQQYJJWDvApMPTxnS3w5v_XXMHQFAvtSxzWpy6E" alt="Logo" width="80" height="80">
  </a>

  <h3 align="center">KUNGFU.AI Project Template</h3>

  <p align="center">
    Your awesome ML project!
    <br />
    <a href=""><strong>Explore the docs »</strong></a>
    <br />
    <br />
    <a href="">View Demo</a>
    ·
    <a href="">Report Bug</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com) -->


### Built With
This section should list any major frameworks that you built your project using. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples.
* [Bootstrap](https://getbootstrap.com)
* [JQuery](https://jquery.com)
* [Laravel](https://laravel.com)



<!-- GETTING STARTED -->

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for
development and testing purposes.

Docker is used to ensure consistency across development, test, training, and production
environments. The Docker image is fully self-contained, so all required frameworks and libraries
will be installed as part of the image creation process.

### Install Requirements

Before proceeding, please install the following prerequisites:

- Install [Git](https://git-scm.com)
- Install [Docker](https://www.docker.com) version 18.0 or later
- Install [pre-commit](https://pre-commit.com)
- Install [shellcheck](https://www.shellcheck.net/)

Easy install for MacOS

```shell script
brew install git
brew install docker
brew install pre-commit
brew install shellcheck
```

---
### Clone the Repo
Clone the git repository locally and change working directory.

```sh
# clone git repo
$ git clone [REPO_URL]

$ cd [REPO_NAME]
```


### Repo Setup

1. Install pre-commit
2. Run `bin/build.sh` to build the project's Docker image.
```shell script
pre-commit install
bin/build.sh
```

#### Unit Tests

Once the Docker image is built we can run the project's unit tests to verify everything is
working. The `bin/test.sh` script will start a Docker container and execute all unit tests using
the [pytest framework](https://docs.pytest.org/en/latest/).

```sh
# run all tests
$ bin/test.sh
```

By default pytest captures all output sent to `stdout` and `stderr` during test execution. This
can be disabled by passing the `-s` option.

```sh
# run tests with capture disabled and verbose
$ bin/test.sh -s -vv
```

You can see a complete list of test configuration options using `--help`.

#### Interactive Shell

The `bin/shell.sh` script starts a Docker container in interactive mode and drops you into a bash
prompt. This can be useful when using an interactive debugger to step through code.

```sh
# run docker image in interactive bash shell
$ bin/shell.sh
```

### Training

Run the training script to train a new model from scratch.

```sh
$ bin/train.sh
```

### Prediction

Run the prediction script to run a trained model on a dataset.

```sh
$ bin/predict.sh
```

<!-- USAGE EXAMPLES -->
## Usage

In Progress


<!-- ROADMAP -->
## Roadmap

In progress

<!-- CONTRIBUTING -->
## Contributing

Any contributions you make are **greatly appreciated**.

1. Create your Feature Branch (`git checkout -b <your name>/<your feature name>`)
2. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
3. Push to the Branch (`git push origin feature/AmazingFeature`)
4. Open a Pull Request



<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact





<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [Img Shields](https://shields.io)
* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Pages](https://pages.github.com)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[license-url]: https://github.com/kungfuai/project-template/blob/master/LICENSE
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/company/kungfuai/
[product-screenshot]: images/screenshot.png
