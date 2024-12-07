# AutoActiveReload_Deadlock
## Introduction
simple computer vision algorithm to automatically detect right timing of pressing R while doing ActiveReload in Deadlock game.

## Feature
- AutoMode: Automatically presses R when reload becomes active.
- Semi-AutoMode: Press R once to schedule automatic input during reload; can be canceled before reload starts. (In progress)
- PreInputMode: Automatically presses R if held by the user when reload becomes active.
- ManualMode: Prevents R input until reload is active.

## Interface Snapshot
![image](https://github.com/user-attachments/assets/302799a9-8e94-4216-86c7-618c8bbb7800)

## Weakness and TODO
- Algorthim Optimization: Currently in some extreme cases will cause script misjudgment.
- Semi-Automode Refinement: Currently this mode could be unavaliable. 
- More resolution support: Currently only support 4K, 2k_16_10 and 2K_16_9.
- English support: As literal:)
