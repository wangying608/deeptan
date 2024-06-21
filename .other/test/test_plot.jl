## Plot
# cd("/home/malab20/_wuchenh/new/test")
include("test_pipe.jl")


findmax(filter(!isnan, MI_omics_mat[:,23]))

MI_omics_vec = filter(!isnan, [x for x in MI_omics_mat])
findmax(MI_omics_vec)

samples_omics[999, 1]
whsamp = samples_omics[999, 2]

x1, y1 = omics_mat[:, 1], omics_mat[:, 1000]
scatter(x1, y1; color=(:blue, 0.3),
    figure = (; resolution = (1080, 1080)))
#
scatter(x1[whsamp], y1[whsamp]; color=(:blue, 0.3),
    figure = (; resolution = (1080, 1080)))
#
hist(MI_omics_mat[]; bins=4)

#
hexbin(x1, y1;
        bins=75,
        cellsize=0.05,
        colormap = :heat,#[Makie.to_color(:transparent); Makie.to_colormap(:viridis)],
        figure = (; resolution = (1080, 1080)))
# save("plot_MI_.svg", scene_, pt_per_unit = 0.5)

#= #----------------------------------------------------


=# #----------------------------------------------------


#
hist(filter(!isnan, MI_omics_mat[:,1]); bins=64,
    figure = (; resolution = (1200, 800)),
    axis = (; title = "Distribution of MI of f1", xlabel = "Mutual information (MI)", ylabel = "Frequency"))
#
lines(1:3999, sort(filter(!isnan, MI_omics_mat[:,1]), rev=true))
lines(1:3999, sumSortedVec(filter(!isnan, MI_omics_mat[:,1]), false))
#


# findall(x -> x > 1.5, filter(!isnan, MI_omics_mat[:,1]))
# findall(x -> x < 0.1, filter(!isnan, MI_omics_mat[:,1]))







#=
x1, y1 = rm_outliers_2vec(mrna[:,1], meth[:,1])
scatter(x1, y1;
    figure = (; resolution = (1080, 1080)),
    axis = (; title = "My Test", xlabel = "mRNA 1", ylabel = "meth 1"))
#
#
x987, y3000 = rm_outliers_2vec(mrna[:,987], meth[:,3000])
scatter(x987, y3000;
    figure = (; resolution = (1080, 1080)),
    axis = (; title = "My Test", xlabel = "mRNA 987", ylabel = "meth 3000"))
#
#
MI_xy(mrna[:,686], meth[:,1832])
x686, y1832 = rm_outliers_2vec(mrna[:,686], meth[:,1832])
scatter(x686, y1832;
    figure = (; resolution = (1080, 1080)),
    axis = (; title = "My Test", xlabel = "mRNA 686", ylabel = "meth 1832"))
#


findmin(dep2om)# : (0.0001361396960793959, CartesianIndex(562, 1054))
findmax(dep2om)# : (3.610208686603059, CartesianIndex(738, 342))
#### Plot x & y
x738, y342 = rm_outliers_2vec(mrna[:,738], meth[:,342])
scatter(x738, y342; color=(:blue, 0.4),
    figure = (; resolution = (1080, 1080)),
    axis = (; title = "My Test", xlabel = "mRNA 738", ylabel = "meth 342"))
=#


#=
findmax(cor2m_pearson)# : (0.7458080660944415, CartesianIndex(882, 860))
findmax(cor2m_spearman)# : (0.824703023510867, CartesianIndex(882, 860))
scatter(mrna[:,882], meth[:,860];
    figure = (; resolution = (800, 800)),
    axis = (; title = "My Test", xlabel = "mRNA F882", ylabel = "meth F860"))

findmin(cor2m_x123)# : (8.196214819540817e-5, CartesianIndex(686, 1832))
#### Plot x & y
xx, yy = rm_outliers_2vec(mrna[:,686], meth[:,1832])
scatter(xx, yy;
    figure = (; resolution = (800, 800)),
    axis = (; title = "My Test", xlabel = "mRNA F686", ylabel = "meth F1832"))
#### Plot y & yhat
pcc1, lnregreMd = auto_regres(xx, yy)
yyhat = predict(lnregreMd, DataFrame(x=xx))
cor(yy, yyhat)
scatter(yy, yyhat;
    figure = (; resolution = (800, 800)),
    axis = (; title = "My Test", xlabel = "meth F1054", ylabel = "meth F1054 pred"))
=#
