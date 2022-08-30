

library(shiny)
library(shinydashboard)
library(ggplot2)
library(curl)
require(biclust)
library(caret)
library(MASS)
library(dplyr)
library(data.table)
library(Matrix)
library(MLmetrics)
library(lightgbm)
library(stats)
library(latex2exp)
library(ggtext) 

dm_all = read.csv("/Volumes/JasonWork/UKB/R_Deploy/Deploy_Models/Calibrations/dm_all.csv")
dm_all$deciles <- as.factor(as.vector(dm_all$deciles))

ad_all = read.csv("/Volumes/JasonWork/UKB/R_Deploy/Deploy_Models/Calibrations/ad_all.csv")
ad_all$deciles <- as.factor(as.vector(ad_all$deciles))


lgb_dir = "https://raw.githubusercontent.com/jasonHKU0907/UKB_DRP/main/Deploy_Models/lgbm_models/"
iso_dir = "https://raw.githubusercontent.com/jasonHKU0907/UKB_DRP/main/Deploy_Models/isoreg_models/"

load_cv_mods = function(mod_dir, mod_name){
  model_lst = list()
  if(grepl("lgb", mod_name, fixed = TRUE) == TRUE){
    for(i in 1:5){
      model_lst[[i]] = readRDS.lgb.Booster(gzcon(url(paste0(mod_dir, mod_name, as.character(i),".rds"))))
    }
  }
  else{
    for(i in 1:5){
      model_lst[[i]] = readRDS(gzcon(url(paste0(mod_dir, mod_name, as.character(i),".rds"))))
    }
  }
  return (model_lst)
}


lgb_lst = load_cv_mods(lgb_dir, "lgb_mod")
iso_dm_all_lst = load_cv_mods(iso_dir, "dm_all/iso_mod")
iso_dm_10yrs_lst = load_cv_mods(iso_dir, "dm_10yrs/iso_mod")
iso_dm_5yrs_lst = load_cv_mods(iso_dir, "dm_5yrs/iso_mod")
iso_ad_all_lst = load_cv_mods(iso_dir, "ad_all/iso_mod")
iso_ad_10yrs_lst = load_cv_mods(iso_dir, "ad_10yrs/iso_mod")
iso_ad_5yrs_lst = load_cv_mods(iso_dir, "ad_5yrs/iso_mod")



ui <- dashboardPage(
  dashboardHeader(title = "UKB-DRP Tool"),
  dashboardSidebar(width = 550,
                   div(style = "font-size: 14px;
                   padding: 0px 0px;
                   margin-top:.95cm;
                       margin-left:0.5cm",
                       fluidRow(column(6, sliderInput("age", h4("Age (years)"), min = 40, max = 90, value = 60, step = 1)),
                                column(5, numericInput("pmtime", h4("Pairs matching time (seconds)"), value = 400, step = 10)))),
                   
                   div(style = "font-size: 14px;
                   padding: 0px 0px;
                   margin-top:.6cm;
                       margin-left:0.5cm",
                       fluidRow(column(6, radioButtons("apoe4", h4("ApoE4 carrier"),
                                                       choices = list("Non carrier" = 0, 
                                                                      "Single-copy carrier" = 1,
                                                                      "Double-copies carrier" = 2), selected = 0)),
                                column(5, numericInput("reatime", h4("Reaction time (seconds)"),value = 550, step = 10)))),
                   
                   div(style = "font-size: 14px;
                   padding: 0px 0px;
                   margin-top:.6cm;
                       margin-left:0.5cm",
                       fluidRow(column(6, sliderInput("nbmed", h4("Number of medications"), min = 0, max = 10, value = 2, step = 1)),
                                column(5, numericInput("legfat", h4("Leg fat percentage (%)"),value = 30, step = 0.5)))),
                   
                   div(style = "font-size: 14px;
                   padding: 0px 0px;
                   margin-top:.6cm;
                       margin-left:0.5cm",
                       fluidRow(column(6, radioButtons("lsill", h4("Long-standing illness"), 
                                                       choices = list("No" = 0, 
                                                                      "Yes" = 1), selected=0)),
                                column(5, numericInput("pef", h4("Peak expiratory flow (L/min)"), value = 300, step=10)))),
                   
                   div(style = "font-size: 14px;
                   padding: 0px 0px;
                   margin-top:.6cm;
                       margin-left:0.5cm",
                       fluidRow(column(6, sliderInput("mda", h4("Mother's age at death (years) (select 100 if still alive)"), 
                                                      min = 30, max = 100, value = 100, step = 1)), 
                                column(5, numericInput("mcv", h4("Mean corpuscular volume (fL)"), value = 90, step = 0.5)))),
                   
                   div(style = "font-size: 14px;
                   padding: 0px 0px;
                   margin-top:2.0cm;
                   margin-left:.85cm",
                       fluidRow(column(11, h5("The UK Biobank-Dementia Risk Prediction (UKB-DRP) tool was established 
                                          based on UK Biobank study cohort. The tool was developed on research purpose
                                          and cannot be used as clinical evidence."))))
  ),
  
  dashboardBody(
    div(style="text-align:center;
                    position:relative;
                    font-size: 30px;
                    font-family: 'Times New Roman', Times, serif;
                    margin-bottom:10px",
        textOutput("title1")),
    div(style="text-align:center;
                    position:relative;
                    margin-bottom:30px",
        plotOutput("plot1", width = "100%")),
    div(style="text-align:center;
                    position:relative;
                    font-size: 30px;
                    font-family: 'Times New Roman', Times, serif;
                    margin-bottom:10px",
        textOutput("title2")),
    div(style="text-align:center;
                    position:relative;
                    margin-bottom:10px",
        plotOutput("plot2", width = "100%"))
  )
)


server <- function(input, output) {
  output$plot1<-renderPlot({
    X_test = as.data.frame(cbind(input$age, input$apoe4, input$pmtime, input$legfat, input$nbmed, 
                                 input$reatime, input$pef, input$mda, input$lsill, input$mcv))
    colnames(X_test) = c('age', 'apoe4', 'pmtime',  'legfat', 'nbmed',
                         'reatime', 'pef', 'mda', 'lsill', 'mcv')
    X_test$age = as.numeric(X_test$age)
    X_test$apoe4 = as.numeric(X_test$apoe4)
    X_test$pmtime = as.numeric(X_test$pmtime)
    X_test$legfat = as.numeric(X_test$legfat)
    X_test$nbmed = as.numeric(X_test$nbmed)
    X_test$reatime = as.numeric(X_test$reatime)
    X_test$pef = as.numeric(X_test$pef)
    X_test$mda = as.numeric(X_test$mda)
    X_test$lsill = as.numeric(X_test$lsill)
    X_test$mcv = as.numeric(X_test$mcv)
    X_test$mda = ifelse(X_test$mda == 100, NA, X_test$mda)
    X_test = as.matrix(X_test[,1:10])
    pred_probs_df = as.data.frame(matrix(0, 5, 7))
    colnames(pred_probs_df) = c('pred_raw', 
                                'pred_dm_all', 
                                'pred_dm_10yrs', 
                                'pred_dm_5yrs')
    for(i in 1:5){
      lgb_model = lgb_lst[[i]]
      iso_dm_all = iso_dm_all_lst[[i]]
      iso_dm_10yrs = iso_dm_10yrs_lst[[i]]
      iso_dm_5yrs = iso_dm_5yrs_lst[[i]]
      pred_prob_raw = predict(lgb_model, X_test)
      pred_probs_df$pred_raw[i] = pred_prob_raw
      pred_probs_df$pred_dm_all[i] = iso_dm_all(pred_prob_raw)
      pred_probs_df$pred_dm_10yrs[i] = iso_dm_10yrs(pred_prob_raw)
      pred_probs_df$pred_dm_5yrs[i] = iso_dm_5yrs(pred_prob_raw)
    }
    df_mean = apply(pred_probs_df, 2, mean)
    df_sd = apply(pred_probs_df, 2, sd)
    df_lbd = df_mean - 1.96*df_sd
    df_ubd = df_mean + 1.96*df_sd
    df_mean1 = df_mean*1000
    lbd_mean1 = df_lbd*1000
    ubd_mean1 = df_ubd*1000
    ylim_ubd = ifelse(ubd_mean1[2]>65, ubd_mean1[2], 65)
    mytitle = paste0("The risk to become dementia (> 10 years): <b style='color:red'>", 
                     as.character(round(df_mean1[2],2)),"\u2030 </b>")
    mysubtitle = paste("10-year risk: <b style='color:red'>", as.character(round(df_mean1[3],2)),
                       "\u2030 </b>  \n 5-year risk: <b style='color:red'>",
                       as.character(round(df_mean1[4],2)), "\u2030 </b>")
    mycaption = expression("Incidence of dementia events in derivational cohort : 12.4\u2030
                           \n Model AUC: 0.848 \u00B1 0.007")
    ggplot(data=dm_all, aes(x=deciles, y=props, fill=groups)) +
      geom_bar(stat="identity", position=position_dodge())+
      geom_text(aes(label=props), vjust=-1, color="black",
                position = position_dodge(1), size=5.5, family = 'Times')+
      scale_fill_manual("legend", values = c("observed" = "steelblue4", "predicted" = "orange2")) +
      ylim(0, ylim_ubd) +
      labs(title = mytitle,
           subtitle = mysubtitle,
           caption = mycaption,
           x = 'Decile groups (10% quantile each)',
           y = "Frequency (\u2030)")+
      theme(legend.position = c(0.1, 0.8), 
            legend.title = element_blank(),
            legend.text = element_text(size=16, face="bold", family = 'Times'),
            legend.spacing.y = unit(-0.05, 'cm'),
            legend.spacing.x = unit(0.5, 'cm'),
            legend.key.size = unit(1, "cm"),
            legend.background = element_rect(fill="lightgray", size=0.8, 
                                             linetype="solid", colour ="darkblue"))+
      theme(axis.title.x = element_text(size=22, family = 'Times'),
            axis.text.x = element_text(size=14, family = 'Times'),
            axis.title.y = element_text(size=20, family = 'Times'),
            axis.text.y = element_text(size=14, family = 'Times'),
            plot.title = element_markdown(size = 22, lineheight = 1.5, family = 'Times'),
            plot.subtitle = element_markdown(size = 20, lineheight = 1.2, family = 'Times'),
            plot.caption = element_markdown(size = 12, lineheight = 1, hjust = 1, family = 'Times'))+
      theme(aspect.ratio = 0.225, plot.margin = unit(c(0.5,0.8,0.5,.8), "cm"))+
      geom_hline(yintercept=df_mean1[2], linetype="dashed", color = "red", size=1.0)
  })
  
  
  
  output$plot2<-renderPlot({
    X_test = as.data.frame(cbind(input$age, input$apoe4, input$pmtime, input$legfat, input$nbmed, 
                                 input$reatime, input$pef, input$mda, input$lsill, input$mcv))
    colnames(X_test) = c('age', 'apoe4', 'pmtime',  'legfat', 'nbmed',
                         'reatime', 'pef', 'mda', 'lsill', 'mcv')
    X_test$age = as.numeric(X_test$age)
    X_test$apoe4 = as.numeric(X_test$apoe4)
    X_test$pmtime = as.numeric(X_test$pmtime)
    X_test$legfat = as.numeric(X_test$legfat)
    X_test$nbmed = as.numeric(X_test$nbmed)
    X_test$reatime = as.numeric(X_test$reatime)
    X_test$pef = as.numeric(X_test$pef)
    X_test$mda = as.numeric(X_test$mda)
    X_test$lsill = as.numeric(X_test$lsill)
    X_test$mcv = as.numeric(X_test$mcv)
    X_test$mda = ifelse(X_test$mda == 100, NA, X_test$mda)
    X_test = as.matrix(X_test[,1:10])
    pred_probs_df = as.data.frame(matrix(0, 5, 7))
    colnames(pred_probs_df) = c('pred_raw', 
                                'pred_ad_all', 
                                'pred_ad_10yrs', 
                                'pred_ad_5yrs')
    for(i in 1:5){
      lgb_model = lgb_lst[[i]]
      iso_ad_all = iso_ad_all_lst[[i]]
      iso_ad_10yrs = iso_ad_10yrs_lst[[i]]
      iso_ad_5yrs = iso_ad_5yrs_lst[[i]]
      pred_prob_raw = predict(lgb_model, X_test)
      pred_probs_df$pred_raw[i] = pred_prob_raw
      pred_probs_df$pred_ad_all[i] = iso_ad_all(pred_prob_raw)
      pred_probs_df$pred_ad_10yrs[i] = iso_ad_10yrs(pred_prob_raw)
      pred_probs_df$pred_ad_5yrs[i] = iso_ad_5yrs(pred_prob_raw)
    }
    df_mean = apply(pred_probs_df, 2, mean)
    df_sd = apply(pred_probs_df, 2, sd)
    df_lbd = df_mean - 1.96*df_sd
    df_ubd = df_mean + 1.96*df_sd
    df_mean1 = df_mean*1000
    lbd_mean1 = df_lbd*1000
    ubd_mean1 = df_ubd*1000
    mytitle = paste0("The risk to become AD (> 10 years): <b style='color:red'>", 
                     as.character(round(df_mean1[2],2)),"\u2030 </b>")
    mysubtitle = paste("10-year risk: <b style='color:red'>", as.character(round(df_mean1[3],2)),
                       "\u2030 </b>  \n 5-year risk: <b style='color:red'>", 
                       as.character(round(df_mean1[4],2)), "\u2030 </b>")
    mycaption = expression("Incidence of AD events in derivational cohort : 5.7\u2030
                           \n Model AUC: 0.862 \u00B1 0.015")    
    ylim_ubd = ifelse(ubd_mean1[2]>35, ubd_mean1[2], 35)
    ggplot(data=ad_all, aes(x=deciles, y=props, fill=groups)) +
      geom_bar(stat="identity", position=position_dodge())+
      geom_text(aes(label=props), vjust=-1, color="black",
                position = position_dodge(1), size=5.5, family = 'Times')+
      scale_fill_manual("legend", values = c("observed" = "steelblue4", "predicted" = "orange2")) +
      ylim(0, ylim_ubd) +
      labs(title = mytitle,
           subtitle = mysubtitle,
           caption = mycaption,
           x = 'Decile groups (10% quantile each)',
           y = "Frequency (\u2030)")+
      theme(legend.position = c(0.1, 0.8), 
            legend.title = element_blank(),
            legend.text = element_text(size=16, face="bold", family = 'Times'),
            legend.spacing.y = unit(-0.05, 'cm'),
            legend.spacing.x = unit(0.5, 'cm'),
            legend.key.size = unit(1, "cm"),
            legend.background = element_rect(fill="lightgray", size=0.8, 
                                             linetype="solid", colour ="darkblue"))+
      theme(axis.title.x = element_text(size=22, family = 'Times'),
            axis.text.x = element_text(size=14, family = 'Times'),
            axis.title.y = element_text(size=20, family = 'Times'),
            axis.text.y = element_text(size=14, family = 'Times'),
            plot.title = element_markdown(size = 22, lineheight = 1.5, family = 'Times'),
            plot.subtitle = element_markdown(size = 20, lineheight = 1.2, family = 'Times'),
            plot.caption = element_markdown(size = 12, lineheight = 1, hjust = 1, family = 'Times'))+
      theme(aspect.ratio = 0.225, plot.margin = unit(c(0.5,0.8,0.5,.8), "cm"))+
      geom_hline(yintercept=df_mean1[2], linetype="dashed", color = "red", size=1.0)
  })
  
  
  
  output$title1 <- renderText({
    print("Risk group stratifications of dementia")
  })
  
  output$title2 <- renderText({
    print("Risk group stratifications of Alzheimer's Disease")
  })
  
}

shinyApp(ui, server)

