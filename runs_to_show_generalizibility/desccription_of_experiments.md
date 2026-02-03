
## Experimental Parameters

Four distinct RDE configurations were investigated to demonstrate the generalizability of the model. The \textbf{one detonation} case features a single detonation wave with 16 injection slots operating at an equivalence ratio of $\phi = 1.0$ and injection pressure of $P_{\text{inj}} = 12$ atm. The \textbf{pulsing} configuration maintains similar injection parameters ($\phi = 1.0$, $P_{\text{inj}} = 12$ atm, 16 slots) but exhibits temporal oscillations in the detonation wave structure. The \textbf{two corotating} case demonstrates a dual-wave mode with two detonation waves rotating in the same direction, using 32 injection slots at $\phi = 0.8$ and $P_{\text{inj}} = 10$ atm. Finally, the \textbf{four corotating} configuration represents the most complex mode with four simultaneously rotating detonation waves, employing 64 injection slots at $\phi = 0.6$ and $P_{\text{inj}} = 8$ atm. These cases span a range of operational regimes from single-wave to multi-wave dynamics, providing a comprehensive test of the model's capability to capture diverse RDE phenomena.

## Comparative Visualization

```latex
\begin{figure}[htbp]
    \centering
    \begin{subfigure}[b]{0.24\textwidth}
        \includegraphics[width=\textwidth]{runs_to_show_generalizibility/one_detonation/result.png}
        \caption{One Detonation}
        \label{fig:one_det}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.24\textwidth}
        \includegraphics[width=\textwidth]{runs_to_show_generalizibility/pulsing/result.png}
        \caption{Pulsing}
        \label{fig:pulsing}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.24\textwidth}
        \includegraphics[width=\textwidth]{runs_to_show_generalizibility/two_corotating/result.png}
        \caption{Two Corotating}
        \label{fig:two_corot}
    \end{subfigure}
    \hfill
    \begin{subfigure}[b]{0.24\textwidth}
        \includegraphics[width=\textwidth]{runs_to_show_generalizibility/four_corotating/result.png}
        \caption{Four Corotating}
        \label{fig:four_corot}
    \end{subfigure}
    \caption{Comparison of four RDE operational modes demonstrating model generalizability across different wave structures and injection configurations.}
    \label{fig:rde_comparison}
\end{figure}
```


## Parameters for sim on which we assimilate

We find a set of parameters which best resemble the data observed in the high resolution 3d simulation. For the following set of parameters we are able to achieve a quasi-steady state of three corortating waves. 
```latex
\begin{table}[htbp]
    \centering
    \begin{tabular}{ll}
        \toprule
        \textbf{Parameter} & \textbf{Value} \\
        \midrule
        $\gamma$ & 1.29 \\
        $p_{\text{ref}}$ & 1.0 \\
        $\rho_{\text{ref}}$ & 1.0 \\
        $R$ & 1.0 \\
        $T_{\text{ref}}$ & 1.0 \\
        $AR$ & 0.2 \\
        $s$ & 0.07 \\
        $\beta$ & 14.286 \\
        $Da$ & 289 \\
        $T_{\text{ign}}$ & 5.8 \\
        $h_v$ & 24.6 \\
        $E_a$ & 11.5 \\
        $L$ & 24.0 \\
        $m_x$ & 100 \\
        $t_{\text{final}}$ & 180.0 \\
        \bottomrule
    \end{tabular}
    \caption{Simulation parameters for three corotating waves configuration.}
    \label{tab:assim_params}
\end{table}
```