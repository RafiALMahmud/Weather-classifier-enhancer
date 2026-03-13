# SSH Push Guide

Use this if you want to push the prepared files to GitHub from your own machine.

## 1) Open terminal in the project folder

```bash
cd Weather-classifier-enhancer
```

## 2) Initialize git if needed

```bash
git init
```

## 3) Add your GitHub repo as origin

```bash
git remote add origin git@github.com:RafiALMahmud/Weather-classifier-enhancer.git
```

If origin already exists:

```bash
git remote set-url origin git@github.com:RafiALMahmud/Weather-classifier-enhancer.git
```

## 4) Generate an SSH key if you do not already have one

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

Press Enter through the prompts, or set a passphrase if you want.

## 5) Start ssh-agent and add the key

### Linux / macOS
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

### Git Bash on Windows
```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_ed25519
```

## 6) Copy the public key

```bash
cat ~/.ssh/id_ed25519.pub
```

Copy the output and add it to GitHub:
- GitHub Settings
- SSH and GPG keys
- New SSH key

## 7) Test SSH

```bash
ssh -T git@github.com
```

## 8) Commit and push

```bash
git add .
git commit -m "Initial commit: add weather classifier pipeline, GUI, docs, and CI workflow"
git branch -M main
git push -u origin main
```

## If you prefer HTTPS instead of SSH

```bash
git remote set-url origin https://github.com/RafiALMahmud/Weather-classifier-enhancer.git
```

Then push normally:

```bash
git add .
git commit -m "Initial commit"
git branch -M main
git push -u origin main
```

GitHub will then ask for authentication through your configured credential manager or token flow.
