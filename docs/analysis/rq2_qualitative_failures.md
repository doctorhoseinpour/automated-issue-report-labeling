# RQ2.8 — Qualitative failure samples

Issues where all four Qwen sizes' best plain RAGTAG (agnostic) predicted **bug** despite ground truth being **feature** or **question**.

Total consensus-bug failures: **264** (of 3300 test issues = 8.0%).

| # | GT | Auto-cat | Title | Body excerpt |
|---:|---|---|---|---|
| 1 | feature | ambiguous | Go to next change doesn't show in command palette in diff editors | Steps to Reproduce:  1. Open a git change in the scm view 2. Try using `Move to Next Change` in the command to navigate to a diff in the file   **Bug*... |
| 2 | feature | ambiguous | Organize Imports doesn't treat exports the same way and merges groups | <!-- ⚠️⚠️ Do Not Delete This! bug_report_template ⚠️⚠️ -->  <!-- Please read our Rules of Conduct: https://opensource.microsoft.com/codeofconduct/ -->... |
| 3 | feature | ambiguous | Settings editor doesn't respect ctrl+up and ctrl+down | Testing #188536    Not sure if the settings editor is intended to support this, but it seems very similar to the other places that do support it.    ... |
| 4 | feature | ambiguous | Arrows to navigate to next and previous months are not announcing properly | Copied from internal bug b/236668243:    When the arrow button for next month is tapped, the button will be announcing the current month. We are expec... |
| 5 | feature | ambiguous | IDE0031 codefix shouldn't remove existing braces | ![image](https://user-images.githubusercontent.com/31348972/208076666-66c6945f-737d-44a5-9bd0-574fb5f3a670.png)    Repro: https://github.com/Youssef13... |
| 6 | feature | ambiguous | [stable-2.13] Fix misrendered sections in manpage generation | Backport of https://github.com/ansible/ansible/pull/80450    This change fixes bugs in the manpage generator that existed since it was first added.   ... |
| 7 | feature | ambiguous | Contextual type for an array element doesn't filter out non-array union members | # Bug Report    ### 🔎 Search Terms    array tuple contextual type inference union implicit any    ### 🕗 Version & Regression Information    - This is ... |
| 8 | feature | ambiguous | DOM renderer does not show selection over regular background colors | When gpuAcceleration = 'off', the yellow text background should be blue here:  ![image](https://user-images.githubusercontent.com/2193314/188175658-0b... |
| 9 | feature | ambiguous | testing.openTesting is not working for testing explorer now | Version: 1.82.0-insider (user setup)  Commit: a0377f0c51dbb2d3188565cdf35e89929f864e65  Date: 2023-08-24T05:32:35.024Z  Electron: 25.5.0  ElectronBuil... |
| 10 | feature | ambiguous | "Add missing properties" quick fix for calculated string values generate either  | # Bug Report    The quick fix "Add missing properties" on objects with inferred/calculated strings produces either `undefined` or `""` values even the... |
| 11 | feature | ambiguous | [analyzer] ConvertToInitializingFormal doesn't handle optional parameters | ```dart  class A {    A({String? s}) {      this.s = s;    }    String? s;  }  ```    or  ```dart  class A {    A([String? s]) {      this.s = s;    }... |
| 12 | feature | ambiguous | CS1591 - False positive for interface-implementing member | [CS1591](https://learn.microsoft.com/en-us/dotnet/csharp/language-reference/compiler-messages/cs1591) - `Missing XML comment for publicly visible type... |
| 13 | feature | ambiguous | Automated cherry pick of #119835: Avoid returning nil responseKind in v1beta1 ag | Cherry pick of #119835 on release-1.27.    #119835: Avoid returning nil responseKind in v1beta1 aggregated    For details on the cherry pick process, ... |
| 14 | feature | ambiguous | Bug: react-devtools not working inside react based chrome extensions | react-devtools not working inside react based chrome extensions    React version: 17.0.1    ## Steps To Reproduce    1. install react-devtools extensi... |
| 15 | feature | question-phrased | "Object literal may only specify known properties" doesn't work on arrays | # Bug Report    <!--    Please fill in each section completely. Thank you!  -->    ### 🔎 Search Terms    <!--    What search terms did you use when tr... |
| 16 | question | ambiguous | main.cpp error during make. Ubuntu 16.04 with boost 1.77 | <!-- This issue tracker is only for technical issues related to Bitcoin Core.    General bitcoin questions and/or support requests are best directed t... |
| 17 | question | ambiguous | Record's default ToString() outputs private getter value | ### Description  Even though `record`'s default `ToString()` implementation doesn't emit any non-`public` properties/fields, a `public` property with ... |
| 18 | question | error-trace | casting with `as` doesn't work on `List`s | ```dart  void main() {    foo([1]);  }    void foo(List<Object> value) {    value as List<int>;  }  ```    ```  TypeError: Instance of 'JSArray<Object... |
| 19 | question | ambiguous | failed_when with multiple conditions not working | ### Summary  i tested with " with ' without " without ' with ( ) without () and in any other combinations you can imagine  ### Issue Type  Bug Report ... |
| 20 | question | ambiguous | Pods/OpenCV2_iOS/opencv2.framework/opencv2(opencl_kernels_calib3d.o), building f | ### System Information  OpenCV version: 4.3.0  Operating System / Platform: Apple M1 Pro / iOS  Xcode Version - 14.3.1  ### Detailed description  Afte... |
| 21 | question | ambiguous | Generics are assumed to be nullable types and lose nullable annotations | When annotating an unconstrained generic T as nullable (T?) the compiler treats this as equivalent to T which is only true if it was already a nullabl... |
| 22 | question | ambiguous | [Web][Channel beta] Unable to paint new Scrollbar | I really can't find a way to change the default color of the new Scrollbar, I can change the Thumb color using the RawScrollbar class but not the trac... |
| 23 | question | ambiguous | [Flutter web]: `removeEventListener` on HTML nodes does not removed actually | I need a functionality where the default context menu is disabled and my own context menu (built by flutter) will show. As described in many `stackove... |
| 24 | question | ambiguous | React DevTools downgrade not working for Chrome. | **Do you want to request a *feature* or report a *bug*?**  Bug  **What is the current behavior?**  The following command fails  ```  yarn run test:chr... |
| 25 | question | ambiguous | BUG in in-built function pow (power function). | Type: <b>Bug</b>    Hello developers hope you are doing well. so while solving my class asignment problem i have found bug in following program as i a... |
| 26 | question | ambiguous | [CS8629]: LINQ .Where calls are not considered | **Version Used**:   .NET 5.0 / but might in .NET 6.0 the same    **Steps to Reproduce**:    ```cs                  IEnumerable<long?> longValues = /* ... |
| 27 | question | ambiguous | same program, different outputs | Type: <b>Bug</b>    #include <iostream>  #include <cmath>  using namespace std;    int main(){      cout<<"Enter a number :";      int n;      cin>>n;... |
| 28 | question | ambiguous | #endif is seen as LeadingTrivia of namespace instead of TrailingTrivia of using  | **Version Used**:     **Steps to Reproduce**:  1. Using statement surrounded by `#if` preprocessor directive  ```  using System;  #if UNITY  using Uni... |
| 29 | question | ambiguous | COLOR_RGB2Lab is not reliable for float32 values | ### System Information    OpenCV python version: 4.8.0.74  Operating System / Platform: Win11  Python version: 3.11 x64    ### Detailed description   ... |
| 30 | question | ambiguous | Bug: React table - Element type is invalid: expected a string (for built-in comp | **Do you want to request a *feature* or report a *bug*?**  bug    **What is the current behavior?**  Hello,  I am new in React and I am trying to crea... |

## Hand-review codebook

- **error-trace**: contains traceback, exception, or *Error* class names — looks bug-like even if it's a question or a feature.
- **question-phrased**: starts with how/what/why/is-it-possible/can-i, or ends with `?` — looks like a question (even if labeled as feature).
- **feature-cue**: phrases like 'feature request', 'proposal', 'add support', 'enhancement' — should have been classified feature.
- **ambiguous**: none of the above — likely labeling noise or genuinely ambiguous.
