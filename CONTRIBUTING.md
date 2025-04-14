# Contributing Guide

Second Me is an open and friendly community. We are dedicated to building a collaborative, inspiring, and exuberant open source community for our members. Everyone is more than welcome to join our community to get help and to contribute to Second Me.

The Second Me community welcomes various forms of contributions, including code, non-code contributions, documentation, and more.

## How to Contribute

| Contribution Type | Details |
|------------------|---------|
| Report a bug | You can file an issue to report a bug with Second Me |
| Contribute code | You can contribute your code by fixing a bug or implementing a feature |
| Code Review | If you are an active contributor or committer of Second Me, you can help us review pull requests |
| Documentation | You can contribute documentation changes by fixing a documentation bug or proposing new content |

## Before Contributing
* Sign [CLA of Mindverse](https://cla-assistant.io/mindverse/Second-Me)
  
## Here is a checklist to prepare and submit your PR (pull request).
* Create your own GitHub branch by forking Second Me
* Checkout [README](README.md) for how to deploy Second Me.
* Push changes to your personal fork.
* Create a PR with a detailed description, if commit messages do not express themselves.
* Submit PR for review and address all feedbacks.
* Wait for merging (done by committers).

## Branch Management Strategy

We follow a structured branching strategy to manage releases and contributions from both internal and external contributors.

### Branch Structure

```
master (stable version)
    ^
    |
release/vX.Y.Z (release preparation branch)
    ^
    |
develop (development integration branch)
    ^
    |
feature/* (feature branches) / hotfix/* (hotfix branches)
```

## Development Workflow by Role

### For Internal Developers

```
                 hotfix/fix-bug
                /       \
master ---------+---------+-----+--- ... --> Stable Version
                \               /
                 \             /
release/v1.0 -----+-----------+--- ... --> Release Version
                    \         /
                     \       /
develop --------------+-----+---+--- ... --> Development Integration
                       \       /
                        \     /
feature/new-feature -----+---+--- ... --> Feature Development
```

1. **Create Feature Branch**
   - Create from `develop` branch
   ```bash
   git checkout develop
   git pull
   git checkout -b feature/my-feature-name
   ```

2. **Develop Your Feature**
   - Make changes in your feature branch
   - Commit regularly with descriptive messages

3. **Submit Pull Request**
   - When complete, submit a PR to the `develop` branch
   - For minor bug fixes, you may target the current `release` branch if one exists

4. **Address Review Feedback**
   - Make requested changes and push updates to your branch

### For External Contributors

```
upstream/master (official master branch)
     ^
     |
upstream/develop (official development branch)
     ^
     |
origin/feature/my-feature (your feature branch)
     ^
     |
origin/master (your forked master branch)
```

As an external contributor, you'll need to follow the fork-based workflow. This ensures a safe and organized way to contribute to the project.

### 1. Fork the Repository
1. Visit https://github.com/Mindverse/Second-Me
2. Click the "Fork" button in the top-right corner
3. Select your GitHub account as the destination

### 2. Clone Your Fork
After forking, clone your fork to your local machine:
```bash
cd working_dir
# Replace USERNAME with your GitHub username
git clone git@github.com:USERNAME/Second-Me.git
cd Second-Me
```

### 3. Configure Upstream Remote
To keep your fork up to date with the main repository:
```bash
# Add the upstream repository
git remote add upstream git@github.com:Mindverse/Second-Me.git

# Verify your remotes
git remote -v
# You should see:
# origin    git@github.com:USERNAME/Second-Me.git (fetch)
# origin    git@github.com:USERNAME/Second-Me.git (push)
# upstream  git@github.com:Mindverse/Second-Me.git (fetch)
# upstream  git@github.com:Mindverse/Second-Me.git (push)
```

### 4. Keep Your Fork Updated
Before creating a new feature branch, ensure both your fork's master and develop branches are up to date:
```bash
# Update master branch
git checkout master
git fetch upstream
git rebase upstream/master
git push origin master

# Update develop branch
git checkout develop
git fetch upstream develop
git rebase upstream/develop
git push origin develop
```

### 5. Create a Feature Branch
Always create a new branch from develop for your changes:
```bash
# Make sure you're on develop branch
git checkout develop

# Create and switch to a new branch
git checkout -b feature/your-feature-name
```

### 6. Make Your Changes
Make your desired changes in the feature branch. Make sure to:
- Follow the project's coding style
- Add tests if applicable
- Update documentation as needed

### 7. Commit Your Changes
```bash
# Add your changes
git add <filename>
# Or git add -A for all changes

# Commit with a clear message
git commit -m "feat: add new feature X"
```

### 8. Update Your Feature Branch
Before submitting your PR, update your feature branch with the latest changes:
```bash
# Fetch upstream changes
git fetch upstream

# Rebase your feature branch
git checkout feature/your-feature-name
git rebase upstream/develop
```

### 9. Push to Your Fork
```bash
# Push your feature branch to your fork
git push origin feature/your-feature-name
```

### 10. Create a Pull Request
1. Visit your fork at `https://github.com/USERNAME/Second-Me`
2. Click "Compare & Pull Request"
3. Select:
   - Base repository: `Mindverse/Second-Me`
   - Base branch: `develop` (for new features) or `release/vX.Y.Z` (for bug fixes to upcoming release)
   - Head repository: `USERNAME/Second-Me`
   - Compare branch: `feature/your-feature-name`
4. Fill in the PR template with:
   - Clear description of your changes
   - Any related issues
   - Testing steps if applicable
   - Target version if applicable

### 11. Review Process
1. Maintainers will review your PR
2. Address any feedback by:
   - Making requested changes
   - Pushing new commits to your feature branch
   - The PR will automatically update
3. Once approved, maintainers will merge your PR to the appropriate branch

### 12. CI Checks
- Automated checks will run on your PR
- All checks must pass before merging
- If checks fail, click "Details" to see why
- Fix any issues and push updates to your branch

### For Maintainers

```
                               PR Merge Control Point
                               |
master -------------------------+------ ... --> Stable Version
                               |
                               |
release/v1.0 -----------------+------- ... --> Release Version
                             / |
                            /  |
develop --------------------+--+------ ... --> Development Integration
     ^                     /   |
     |                    /    |
feature/internal --------+     |
     ^                          
     |                          
PR (external) ----------------+
```

1. **Merge PRs**
   - Merge PRs to the appropriate branch
   - Make sure to follow the project's merge strategy

2. **Release Management**
   - Create releases from the `develop` branch
   - Merge releases to both `master` and `develop`
   - Tag releases in `master` with the version number

3. **Hotfixes**
   - Create hotfixes from `master`
   - Merge hotfixes to both `master` and `develop` (and current `release` branch if exists)

## Release Management

### Creating a Release
1. When `develop` branch contains all features planned for a release, a `release/vX.Y.Z` branch is created
2. Only bug fixes and release preparation commits are added to the release branch
3. After thorough testing, the release branch is merged to both `master` and `develop`
4. The release is tagged in `master` with the version number

### Hotfixes
1. For critical bugs in production, create a `hotfix/fix-description` branch from `master`
2. Fix the issue and create a PR targeting `master`
3. After approval, merge to both `master` and `develop` (and current `release` branch if exists)

## Tips for Successful Contributions
- Create focused, single-purpose PRs
- Follow the project's code style and conventions
- Write clear commit messages
- Keep your fork updated to avoid merge conflicts
- Be responsive during the review process
- Ask questions if anything is unclear
