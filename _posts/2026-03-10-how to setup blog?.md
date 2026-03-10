---
title: How to setup my blog?
date: 2026-03-10
categories: [misc, misc_notes]
tags: [misc]
---

# Goal

Setup personal blog on github.io so that I can share my personal experiences about deep learning, large language model, and AI-infra optimization.

# Setup about the blog

To set up this blog, I used the [Chirpy Jekyll theme](https://chirpy.cotes.page/posts/getting-started/) and followed the [Jekyll macOS installation guide](https://jekyllrb.com/docs/installation/macos/). Here is a step-by-step record of the setup process.

## 1. Install Ruby and Jekyll on macOS

macOS comes with a system Ruby, but it's not recommended for development. Instead, I used Homebrew and `chruby` to manage a separate Ruby environment.

First, install Homebrew (if not already installed):
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install `chruby` and `ruby-install`:
```bash
brew install chruby ruby-install
```

Install the latest stable version of Ruby (e.g., 3.4.1):
```bash
ruby-install ruby 3.4.1
```

Configure the shell (e.g., `~/.zshrc`) to automatically use `chruby`:
```bash
echo "source $(brew --prefix)/opt/chruby/share/chruby/chruby.sh" >> ~/.zshrc
echo "source $(brew --prefix)/opt/chruby/share/chruby/auto.sh" >> ~/.zshrc
echo "chruby ruby-3.4.1" >> ~/.zshrc
```

Restart the terminal and install Jekyll:
```bash
gem install jekyll
```

## 2. Create the Site Repository

I used the Chirpy Starter template, which simplifies upgrades and isolates unnecessary files:
1. Navigate to the Chirpy Starter repository on GitHub.
2. Click **Use this template** -> **Create a new repository**.
3. Name the repository `<username>.github.io` (replacing `<username>` with your GitHub username).

## 3. Local Environment Setup

Clone the newly created repository to your local machine:
```bash
git clone https://github.com/<username>/<username>.github.io.git
cd <username>.github.io
```

Install the Ruby dependencies:
```bash
bundle
```

## 4. Run the Local Server

To test the site locally, run the Jekyll server. I used port 4001:
```bash
bundle exec jekyll serve --port 4001
```
The site will be available at `http://127.0.0.1:4001`.

## 5. Deployment via GitHub Actions

Before deploying, there are a few important prerequisites to check:
* **Repository Visibility:** If you are on the GitHub Free plan, ensure your site repository is set to **Public**.
* **Platform Lock:** If you have committed `Gemfile.lock` to your repository and your local machine is not running Linux (e.g., you are on macOS), you must update the platform list of the lock file so GitHub Actions can build it successfully:
  ```bash
  bundle lock --add-platform x86_64-linux
  ```
* **Configuration:** Check your `_config.yml` file and ensure the `url` is configured correctly. If you are not using a custom domain and prefer a project site, remember to set the `baseurl` to your project name starting with a slash (e.g., `/project-name`).

To deploy the site to GitHub Pages:
1. Go to your repository **Settings** on GitHub.
2. Click **Pages** in the left navigation bar.
3. Under the **Build and deployment** section, select **GitHub Actions** as the source from the dropdown menu.
4. Push any commits to GitHub to trigger the *Build and Deploy* workflow. 
5. In the **Actions** tab of your repository, you should see the workflow running. Once the build is complete and successful, the site is live and you can visit the provided URL!

## 6. TODO
Add more tricks for the blog setup and theme.